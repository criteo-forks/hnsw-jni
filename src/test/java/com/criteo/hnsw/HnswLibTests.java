package com.criteo.hnsw;

import org.junit.Ignore;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;

public class HnswLibTests {
    private static float delta = 5.0E-6f;
    private int randomSeed = 42;
    private Random rand = new Random(randomSeed);

    private float seedValue = 1;
    private int dimension = 100;
    private Function<Integer, Float> getValueById = (id) -> id == 0 ? 0 : seedValue / id;
    private Function<Integer, Float> getRandomValueById = (id) -> rand.nextFloat();
    private Function<Integer, Float> getNormalizedValueById = i -> i == 0 ? 0.0f : 1 / (float) Math.sqrt(dimension);
    private long nbItems = 12;
    private int M = 16;
    private int efConstruction = 200;
    private Decoder decoderFloat16 = Decoder.create(dimension, Precision.Float16);
    private Decoder decoderFloat32 = Decoder.create(dimension, Precision.Float32);

    @Test
    public void create_indices_save_and_load() throws IOException {
        Map<HnswIndex, Function<Integer, Float>> indices = new HashMap<HnswIndex, Function<Integer, Float>>() {{
            // Hnsw
            put(HnswIndex.create(Metrics.Euclidean, dimension, Precision.Float32, false), getValueById);
            put(HnswIndex.create(Metrics.DotProduct, dimension, Precision.Float32, false), getValueById);
            put(HnswIndex.create(Metrics.Angular, dimension, Precision.Float32, false), getNormalizedValueById);
            // Bruteforce
            put(HnswIndex.create(Metrics.Euclidean, dimension, Precision.Float32, true), getValueById);
            put(HnswIndex.create(Metrics.DotProduct, dimension, Precision.Float32, true), getValueById);
            put(HnswIndex.create(Metrics.Angular, dimension, Precision.Float32, true), getNormalizedValueById);
        }};
        File dir = Files.createTempDirectory("HnswLib").toFile();
        String indexPathStr = new File(dir, "index.hnsw").toString();

        long nbItems = 123;

        for (Map.Entry<HnswIndex, Function<Integer, Float>> entry : indices.entrySet()) {
            HnswIndex index = entry.getKey();
            Function<Integer, Float> getValueById = entry.getValue();
            if (index.isBruteforce()) {
                index.initBruteforce(nbItems);
            } else {
                index.initNewIndex(nbItems, M, efConstruction, randomSeed);
            }
            assertEquals(0, index.getNbItems());
            populateIndex(index, getValueById, nbItems, dimension);

            assertEquals(nbItems, index.getNbItems());

            assertAllVectorsMatchExpected(index, dimension, getValueById, decoderFloat32, delta);
            index.save(indexPathStr);

            index.load(indexPathStr);
            assertAllVectorsMatchExpected(index, dimension, getValueById, decoderFloat32, delta);

            index.unload();
        }
    }


    @Test
    public void create_indices_save_and_load_float16() throws IOException {
        Map<HnswIndex, Function<Integer, Float>> indices = new HashMap<HnswIndex, Function<Integer, Float>>() {{
            put(HnswIndex.create(Metrics.Euclidean, dimension, Precision.Float16), getValueById);
        }};
        File dir = Files.createTempDirectory("HnswLib").toFile();
        String indexPathStr = new File(dir, "index.hnsw").toString();

        long nbItems = 123;
        double epsilon = 1E-3;
        for (Map.Entry<HnswIndex, Function<Integer, Float>> entry : indices.entrySet()) {
            HnswIndex index = entry.getKey();
            Function<Integer, Float> getValueById = entry.getValue();

            index.initNewIndex(nbItems, M, efConstruction, randomSeed);
            assertEquals(0, index.getNbItems());
            populateIndex(index, getValueById, nbItems, dimension);

            assertEquals(nbItems, index.getNbItems());

            assertAllVectorsMatchExpected(index, dimension, getValueById, decoderFloat16, epsilon);
            index.save(indexPathStr);

            index.load(indexPathStr);
            assertAllVectorsMatchExpected(index, dimension, getValueById, decoderFloat16, epsilon);

            index.unload();
        }
    }


    @Test
    public void check_Euclidean_index_computes_distance_correctly_simple_embeddings() {
        int[] dimensionsToTest = new int[]{3, 6, 16, 101, 200};
        for (int dimension : dimensionsToTest) {
            HnswIndex index = HnswIndex.create(Metrics.Euclidean, dimension, Precision.Float32);

            index.initNewIndex(nbItems, M, efConstruction, randomSeed);
            populateIndex(index, getValueById, nbItems, dimension);

            for (int i = 0; i < nbItems; i++) {
                for (int j = i; j < nbItems; j++) {
                    float distance = index.getDistanceBetweenLabels(i, j);
                    float expectedDistance = dimension * (float) Math.pow((getValueById.apply(i) - getValueById.apply(j)), 2);
                    assertRelativeError(distance, expectedDistance);
                }
            }
            index.unload();
        }
    }

    @Test
    public void check_Euclidean_index_computes_distance_correctly_random_seeded_embeddings() {
        int[] dimensionsToTest = new int[]{3, 6, 16, 101, 200};
        for (int dimension : dimensionsToTest) {
            HnswIndex index = HnswIndex.create(Metrics.Euclidean, dimension, Precision.Float32);

            index.initNewIndex(nbItems, M, efConstruction, randomSeed);
            populateIndex(index, getRandomValueById, nbItems, dimension);

            for (int i = 0; i < nbItems; i++) {
                for (int j = i; j < nbItems; j++) {
                    float distance = index.getDistanceBetweenLabels(i, j);
                    FloatByteBuf vector1 = index.getItem(i);
                    FloatByteBuf vector2 = index.getItem(j);
                    float expectedDistance = getL2Distance(vector1, vector2, dimension);
                    assertRelativeError(distance, expectedDistance);
                }
            }
            index.unload();
        }
    }

    private void assertRelativeError(float distance, float expectedDistance) {
        float error = 0;
        if (expectedDistance != 0) {
            error = Math.abs(expectedDistance - distance) / Math.abs(expectedDistance);
        }
        assertEquals(0, error, delta);
    }

    @Test
    public void check_Euclidean_index_returns_correct_neighbours() throws Exception {
        int k = (int) nbItems;
        HnswIndex[] indices = new HnswIndex[]{
                HnswIndex.create(Metrics.Euclidean, dimension, Precision.Float32, false),
                HnswIndex.create(Metrics.Euclidean, dimension, Precision.Float32, true),
        };

        for (HnswIndex index : indices) {

            if (index.isBruteforce()) {
                index.initBruteforce(nbItems);
            } else {
                index.initNewIndex(nbItems, M, efConstruction, randomSeed);
            }
            populateIndex(index, getValueById, nbItems, dimension);

            FloatByteBuf query = index.getItem(0);
            KnnResult results = index.search(query, k);
            assertEquals(k, results.resultCount);

            // 1st result should be self
            assertEquals(0, results.resultItems[0]);

            assertEquals(0, results.resultDistances[0], delta);
            assertAllEqual(query, results.resultVectors[0], dimension);
            for (int i = 1; i < k; i++) {
                long found = results.resultItems[i];
                float distance = results.resultDistances[i];
                assertEquals(k - i, found);
                assertEquals(dimension * (float) Math.pow(getValueById.apply(k - i), 2), distance, delta);
                assertAllEqual(index.getItem(found), results.resultVectors[i], dimension);
            }

            index.unload();
        }
    }

    @Test
    public void check_Euclidean_float16_index_returns_correct_neighbours() throws Exception {
        int k = (int) nbItems;
        HnswIndex index = HnswIndex.create(Metrics.Euclidean, dimension, Precision.Float16);

        index.initNewIndex(nbItems, M, efConstruction, randomSeed);
        populateIndex(index, getValueById, nbItems, dimension);

        FloatByteBuf query = decoderFloat16.decode(index.getItem(0));
        KnnResult results = index.search(query, k);
        assertEquals(k, results.resultCount);

        // 1st result should be self
        assertEquals(0, results.resultItems[0]);

        assertEquals(0, results.resultDistances[0], 1E-3);
        assertAllEqual(query, decoderFloat16.decode(results.resultVectors[0]), dimension);
        for (int i = 1; i < k; i++) {
            long found = results.resultItems[i];
            float distance = results.resultDistances[i];
            assertEquals(k - i, found);
            assertEquals(dimension * (float) Math.pow(getValueById.apply(k - i), 2), distance, 6E-3);
            assertAllEqual(decoderFloat16.decode(index.getItem(found)), decoderFloat16.decode(results.resultVectors[i]), dimension);
        }

        index.unload();
    }

    @Test
    public void check_no_error_happens_if_less_items_is_returned_than_k() throws Exception {
        int biggerK = (int) nbItems * 2;

        HnswIndex index = HnswIndex.create(Metrics.Euclidean, dimension, Precision.Float32);

        index.initNewIndex(nbItems, M, efConstruction, randomSeed);
        populateIndex(index, getValueById, nbItems, dimension);

        FloatByteBuf query = index.getItem(0);
        KnnResult results = index.search(query, biggerK);
        assertEquals(nbItems, results.resultCount);

        for (int i = 1; i < nbItems; i++) {
            long found = results.resultItems[i];
            float distance = results.resultDistances[i];
            assertEquals(nbItems - i, found);
            assertEquals(dimension * (float) Math.pow(getValueById.apply((int) nbItems - i), 2), distance, delta);
            assertAllEqual(index.getItem(found), results.resultVectors[i], dimension);
        }

        index.unload();
    }


    @Test
    @Ignore("Inner product of A and B is implemented in hnswlib (space_ip.h) as A*B = 1 - (A1*B1 + A2*B2 + â€¦)")
    public void check_DotProduct_index_returns_correct_neighbours() throws Exception {
        int k = (int) nbItems;
        HnswIndex index = HnswIndex.create(Metrics.DotProduct, dimension, Precision.Float32);

        index.initNewIndex(nbItems, M, efConstruction, randomSeed);
        populateIndex(index, getValueById, nbItems, dimension);

        for (int j = 0; j < nbItems; j++) {
            FloatByteBuf query = index.getItem(j);
            KnnResult results = index.search(query, k);

            assertEquals(k, results.resultCount);

            for (int i = 0; i < k; i++) {
                long found = results.resultItems[i];
                float distance = results.resultDistances[i];
                assertEquals(k - i, found);
                assertEquals(1 - dimension * getValueById.apply(k - i) * getValueById.apply(j), distance, delta);
                assertAllEqual(index.getItem(found), results.resultVectors[i], dimension);
            }
        }
        index.unload();
    }

    @Test
    public void check_Kendall_index_computes_distance_correctly() {
        HnswIndex index = HnswIndex.create(Metrics.Kendall, 4, Precision.Float32);
        index.initNewIndex(nbItems, M, efConstruction, randomSeed);
        float[] vector1 = new float[]{1, 2, 3, 4};
        float[] vector2 = new float[]{1, 3, 2, 4};

        index.addItem(vector1, 1);
        index.addItem(vector2, 2);

        float distance = index.getDistanceBetweenLabels(1L, 2L);
        float expectedDistance = (1.0f - 0.66666666f);
        assertRelativeError(distance, expectedDistance);
        index.unload();
    }

    private void populateIndex(HnswIndex index, Function<Integer, Float> getValueById, long nbItems, int dimension) {
        for (int i = 0; i < nbItems; i++) {
            float value = getValueById.apply(i);
            index.addItem(HnswNativeLoadTest.getVector(value, dimension), i);
            index.getItem(i);
        }
    }

    private void assertAllVectorsMatchExpected(HnswIndex index, int size, Function<Integer, Float> getExpectedValue, Decoder decoder, double epsilon) {
        long[] ids = index.getIds();
        for (long l : ids) {
            int id = (int) l;
            float expectedValue = getExpectedValue.apply(id);
            FloatByteBuf item = decoder.decode(index.getItem(id));
            assertAllValuesEqual(item, size, expectedValue, epsilon);
        }
    }

    private void assertAllValuesEqual(FloatByteBuf vector, int size, float expectedValue, double epsilon) {
        assertEquals(size, vector.asFloatBuffer().limit());
        for (int j = 0; j < size; j++) {
            assertEquals(expectedValue, vector.get(j), epsilon);
        }
    }

    private void assertAllEqual(FloatBuffer vector1, FloatBuffer vector2, int size) {
        assertEquals(size, vector2.limit());
        assertEquals(vector1.limit(), vector2.limit());
        for (int j = 0; j < size; j++) {
            assertEquals(vector1.get(j), vector2.get(j), 1E-20);
        }
    }

    private void assertAllEqual(FloatByteBuf vector1, FloatByteBuf vector2, int size) {
        assertAllEqual(vector1.asFloatBuffer(), vector2.asFloatBuffer(), size);
    }

    private float getL2Distance(FloatByteBuf vector1, FloatByteBuf vector2, int dimension) {
        float res = 0.0f;
        for (int i = 0; i < dimension; i++) {
            float t = vector1.get(i) - vector2.get(i);
            res += t * t;
        }
        return res;
    }
}

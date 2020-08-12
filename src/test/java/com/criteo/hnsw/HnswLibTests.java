package com.criteo.hnsw;

import org.junit.Test;

import java.io.File;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;

public class HnswLibTests {
    private static float delta = 8E-6f;
    private int randomSeed = 42;
    private Random rand = new Random(randomSeed);

    private float seedValue = 1;
    private int dimension = 100;
    private Function<Integer, Float> getValueById = (id) -> id == 0 ? 0 : seedValue / id;
    private Function<Integer, Float> getValueByIdFromOne = (id) ->  seedValue / (id + seedValue);
    private Function<Integer, Float> getRandomValueById = (id) -> rand.nextFloat();
    private Function<Integer, Float> getNormalizedValueById = i -> i == 0 ? 0.0f : 1 / (float) Math.sqrt(dimension);
    private long nbItems = 12;
    private int M = 16;
    private int efConstruction = 200;

    @Test
    public void create_indices_save_and_load() throws Exception {
        Map<HnswIndex, Function<Integer, Float>> indices = new HashMap<HnswIndex, Function<Integer, Float>>() {{
            // Hnsw - float32
            put(HnswIndex.create(Metrics.Euclidean, dimension, Precision.Float32, false), getValueById);
            put(HnswIndex.create(Metrics.DotProduct, dimension, Precision.Float32, false), getValueById);
            put(HnswIndex.create(Metrics.Angular, dimension, Precision.Float32, false), getNormalizedValueById);
            // Hnsw - float16
            put(HnswIndex.create(Metrics.Euclidean, dimension, Precision.Float16, false), getValueById);
            put(HnswIndex.create(Metrics.DotProduct, dimension, Precision.Float16, false), getValueById);
            put(HnswIndex.create(Metrics.Angular, dimension, Precision.Float16, false), getNormalizedValueById);
        }};
        File dir = Files.createTempDirectory("HnswLib").toFile();
        String indexPathStr = new File(dir, "index.hnsw").toString();

        long nbItems = 123;

        double epsilon16 = 1E-3;
        double epsilon32 = delta;

        for (Map.Entry<HnswIndex, Function<Integer, Float>> entry : indices.entrySet()) {
            HnswIndex index = entry.getKey();
            Function<Integer, Float> getValueById = entry.getValue();
            index.initNewIndex(nbItems, M, efConstruction, randomSeed);
            assertEquals(0, index.getNbItems());
            populateIndex(index, getValueById, nbItems, dimension);

            assertEquals(nbItems, index.getNbItems());

            double epsilon;
            switch (index.getPrecision()) {
                case (Precision.Float16Val): epsilon = epsilon16; break;
                default: epsilon = epsilon32; break;
            }
            assertAllVectorsMatchExpected(index, dimension, getValueById, epsilon);
            index.save(indexPathStr);

            index.load(indexPathStr);
            assertAllVectorsMatchExpected(index, dimension, getValueById, epsilon);

            // Create float16 Index from either float32 or float16 format
            HnswIndex index16 = HnswIndex.create(index.getMetric(), dimension, Precision.Float16Val, index.isBruteforce());
            index16.load(indexPathStr);
            assertAllVectorsMatchExpected(index16, dimension, getValueById, epsilon16);

            index16.unload();
            index.unload();
        }
    }


    @Test
    public void create_bruteforce_indices_save_and_load() throws Exception {
        Map<HnswIndex, Function<Integer, Float>> indices = new HashMap<HnswIndex, Function<Integer, Float>>() {{
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
            index.initBruteforce(nbItems);
            assertEquals(0, index.getNbItems());
            populateIndex(index, getValueById, nbItems, dimension);

            assertEquals(nbItems, index.getNbItems());

            assertAllVectorsMatchExpected(index, dimension, getValueById, delta);
            index.save(indexPathStr);

            index.load(indexPathStr);
            assertAllVectorsMatchExpected(index, dimension, getValueById, delta);

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
    public void check_Euclidean_bruteforce_search_returns_correct_neighbours() throws Exception {
        int k = (int) nbItems;
        HnswIndex[] indices = new HnswIndex[]{
            HnswIndex.create(Metrics.Euclidean, dimension, Precision.Float32, false),
        };

        for (HnswIndex index : indices) {
            index.initNewIndex(nbItems, M, efConstruction, randomSeed);
            index.enableBruteforceSearch();
            populateIndex(index, getValueById, nbItems, dimension);

            FloatByteBuf query = index.getItem(0);
            KnnResult results = index.searchBruteforce(query, k);
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

        FloatByteBuf query = index.getItemDecoded(0);
        KnnResult results = index.search(query, k);
        assertEquals(k, results.resultCount);

        // 1st result should be self
        assertEquals(0, results.resultItems[0]);
        assertEquals(0, results.resultDistances[0], 1E-3);

        try (FloatByteBuf decoded = index.decode(results.resultVectors[0])) {
            assertAllEqual(query, decoded, dimension);
        }
        for (int i = 1; i < k; i++) {
            long found = results.resultItems[i];
            float distance = results.resultDistances[i];
            assertEquals(k - i, found);
            assertEquals(dimension * (float) Math.pow(getValueById.apply(k - i), 2), distance, 6E-3);
            try(FloatByteBuf decodedFound = index.getItemDecoded(found);
                FloatByteBuf decodedResult = index.decode(results.resultVectors[i])) {
                assertAllEqual(decodedFound, decodedResult, dimension);
            }
        }

        query.close();
        index.unload();
    }

    @Test
    public void check_no_error_happens_if_less_items_is_returned_than_k() throws Exception {
        int biggerK = (int) nbItems * 2;
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
    }


    @Test
    public void check_DotProduct_index_returns_correct_neighbours() throws Exception {
        int k = (int) nbItems;
        HnswIndex index = HnswIndex.create(Metrics.DotProduct, dimension, Precision.Float32);

        index.initNewIndex(nbItems, M, efConstruction, randomSeed);
        populateIndex(index, getValueByIdFromOne, nbItems, dimension);

        FloatByteBuf query = index.getItem(0);
        KnnResult results = index.search(query, k);

        assertEquals(k, results.resultCount);

        // 1st result should be self
        assertEquals(0, results.resultItems[0]);
        assertEquals(1 - dimension, results.resultDistances[0], delta);

        for (int i = 1; i < k; i++) {
            long found = results.resultItems[i];
            float distance = results.resultDistances[i];
            assertEquals(i, found);
            assertEquals(1 - dimension * getValueByIdFromOne.apply(i), distance, delta);
            assertAllEqual(index.getItem(found), results.resultVectors[i], dimension);
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

    private void assertAllVectorsMatchExpected(HnswIndex index, int size, Function<Integer, Float> getExpectedValue, double epsilon) throws Exception {
        long[] ids = index.getIds();
        for (long l : ids) {
            int id = (int) l;
            float expectedValue = getExpectedValue.apply(id);
            System.out.println("Precision: " + Precision.getStr(index.getPrecision()));
            System.out.println("Expected: " + expectedValue + "; label: " + l);
            try(FloatByteBuf item = index.getItemDecoded(id)) {
                assertAllValuesEqual(item, size, expectedValue, epsilon);
            }
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

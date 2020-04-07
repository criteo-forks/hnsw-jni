package com.criteo.hnsw;

import org.junit.Ignore;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
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
    private Function<Integer, Float> getValueById = (id) -> id == 0? 0: seedValue / id;
    private Function<Integer, Float> getRandomValueById = (id) -> rand.nextFloat();
    private Function<Integer, Float> getNormalizedValueById = i -> i == 0? 0.0f: 1/(float)Math.sqrt(dimension);
    private long nbItems = 12;
    private int M = 16;
    private int efConstruction = 200;


    @Test
    public void create_indices_save_and_load() throws IOException {
        Map<Long, Function<Integer, Float>> indices = new HashMap<Long, Function<Integer, Float>>() {{
            put(HnswLib.createEuclidean(dimension), getValueById);
            put(HnswLib.createInnerProduct(dimension), getValueById);
            put(HnswLib.createAngular(dimension), getNormalizedValueById);
        }};

        File dir = Files.createTempDirectory("HnswLib").toFile();
        String indexPathStr = new File(dir, "index.hnsw").toString();

        long nbItems = 123;

        for (Map.Entry<Long, Function<Integer, Float>> entry : indices.entrySet()) {
            long index = entry.getKey();
            Function<Integer, Float> getValueById = entry.getValue();

            HnswLib.initNewIndex(index, nbItems, M, efConstruction, randomSeed);
            assertEquals(0, HnswLib.getNbItems(index));
            populateIndex(index, getValueById, nbItems, dimension);

            assertEquals(nbItems, HnswLib.getNbItems(index));

            assertAllVectorsMatchExpected(index, dimension, getValueById);
            HnswLib.saveIndex(index, indexPathStr);

            HnswLib.loadIndex(index, indexPathStr);
            assertAllVectorsMatchExpected(index, dimension, getValueById);

            HnswLib.destroy(index);
        }
    }


    @Test
    public void check_Euclidean_index_computes_distance_correctly_simple_embeddings() {
        int[] dimensionsToTest = new int[] {3, 6, 16, 101, 200};
        for (int dimension : dimensionsToTest) {
            long index = HnswLib.createEuclidean(dimension);

            HnswLib.initNewIndex(index, nbItems, M, efConstruction, randomSeed);
            populateIndex(index, getValueById, nbItems, dimension);

            for (int i = 0; i < nbItems; i++) {
                for (int j = i; j < nbItems; j++) {
                    float distance = HnswLib.getDistanceBetweenLabels(index, i, j);
                    float expectedDistance = dimension * (float) Math.pow((double) (getValueById.apply(i) - getValueById.apply(j)), 2);
                    assertRelativeError(distance, expectedDistance);
                }
            }
            HnswLib.destroy(index);
        }
    }

    @Test
    public void check_Euclidean_index_computes_distance_correctly_random_seeded_embeddings() {
        int[] dimensionsToTest = new int[] {3, 6, 16, 101, 200};
        for (int dimension : dimensionsToTest) {
            long index = HnswLib.createEuclidean(dimension);

            HnswLib.initNewIndex(index, nbItems, M, efConstruction, randomSeed);
            populateIndex(index, getRandomValueById, nbItems, dimension);

            for (int i = 0; i < nbItems; i++) {
                for (int j = i; j < nbItems; j++) {
                    float distance = HnswLib.getDistanceBetweenLabels(index, i, j);
                    FloatByteBuf vector1 = FloatByteBuf.wrappedBuffer(HnswLib.getItem(index, i));
                    FloatByteBuf vector2 = FloatByteBuf.wrappedBuffer(HnswLib.getItem(index, j));
                    float expectedDistance = getL2Distance(vector1, vector2, dimension);
                    assertRelativeError(distance, expectedDistance);
                }
            }
            HnswLib.destroy(index);
        }
    }

    private void assertRelativeError(float distance, float expectedDistance) {
        float error = 0;
        if(expectedDistance != 0) {
            error = Math.abs(expectedDistance - distance)/Math.abs(expectedDistance);
        }
        assertEquals(0, error, delta);
    }

    @Test
    public void check_Euclidean_index_returns_correct_neighbours() {
        int k = (int)nbItems;
        long index = HnswLib.createEuclidean(dimension);

        HnswLib.initNewIndex(index, nbItems, M, efConstruction, randomSeed);
        populateIndex(index, getValueById, nbItems, dimension);

        FloatByteBuf distances = new FloatByteBuf(k);
        LongByteBuf items = new LongByteBuf(k);
        FloatBuffer query = HnswLib.getItem(index, 0).asFloatBuffer();
        ByteBuffer[] vectors = new ByteBuffer[k];
        long nbResults = HnswLib.search(index, query, k, items.asLongBuffer(), distances.asFloatBuffer(), vectors);
        assertEquals(k, nbResults);

        // 1st result should be self
        assertEquals(0, items.get(0));
        assertEquals(0, distances.get(0), delta);
        assertAllEqual(query, vectors[0].asFloatBuffer(), dimension);
        for(int i = 1; i < k; i++) {
            long found = items.get(i);
            float distance = distances.get(i);
            assertEquals(k - i, found);
            assertEquals(dimension * (float)Math.pow(getValueById.apply(k - i), 2), distance, delta);
            assertAllEqual(HnswLib.getItem(index, found).asFloatBuffer(), vectors[i].asFloatBuffer(), dimension);
        }

        HnswLib.destroy(index);
    }

    @Test
    public void check_no_error_happens_if_less_items_is_returned_than_k() {
        int biggerK = (int)nbItems * 2;

        long index = HnswLib.createEuclidean(dimension);

        HnswLib.initNewIndex(index, nbItems, M, efConstruction, randomSeed);
        populateIndex(index, getValueById, nbItems, dimension);

        FloatByteBuf distances = new FloatByteBuf(biggerK);
        LongByteBuf items = new LongByteBuf(biggerK);
        FloatBuffer query = HnswLib.getItem(index, 0).asFloatBuffer();
        ByteBuffer[] vectors = new ByteBuffer[biggerK];
        long nbResults = HnswLib.search(index, query, biggerK, items.asLongBuffer(), distances.asFloatBuffer(), vectors);

        assertEquals(nbItems, nbResults);

        for(int i = 1; i < nbItems; i++) {
            long found = items.get(i);
            float distance = distances.get(i);
            assertEquals(nbItems - i, found);
            assertEquals(dimension * (float)Math.pow(getValueById.apply((int)nbItems - i), 2), distance, delta);
            assertAllEqual(HnswLib.getItem(index, found).asFloatBuffer(), vectors[i].asFloatBuffer(), dimension);
        }

        HnswLib.destroy(index);
    }


    @Test @Ignore("Inner product of A and B is implemented in hnswlib (space_ip.h) as A*B = 1 - (A1*B1 + A2*B2 + â€¦)")
    public void check_DotProduct_index_returns_correct_neighbours() {
        int k = (int)nbItems;
        long index = HnswLib.createInnerProduct(dimension);

        HnswLib.initNewIndex(index, nbItems, M, efConstruction, randomSeed);
        populateIndex(index, getValueById, nbItems, dimension);

        FloatByteBuf distances = new FloatByteBuf(k);
        FloatBuffer distancesNio = distances.asFloatBuffer();
        LongByteBuf items = new LongByteBuf(k);
        LongBuffer itemsNio = items.asLongBuffer();
        ByteBuffer[] vectors = new ByteBuffer[k];
        for(int j = 0; j < nbItems; j++) {
            FloatBuffer query = HnswLib.getItem(index, j).asFloatBuffer();
            HnswLib.search(index, query, k, itemsNio, distancesNio, vectors);

            assertEquals(k, distancesNio.limit());
            assertEquals(k, itemsNio.limit());

            for (int i = 0; i < k; i++) {
                long found = items.get(i);
                float distance = distances.get(i);
                assertEquals(k - i, found);
                assertEquals(1 - dimension * getValueById.apply(k - i) * getValueById.apply(j), distance, delta);
                assertAllEqual(HnswLib.getItem(index, found).asFloatBuffer(), vectors[i].asFloatBuffer(), dimension);
            }
        }
        HnswLib.destroy(index);
    }

    @Test
    public void check_Kendall_index_computes_distance_correctly() {
        long index = HnswLib.createKendall(4);
        HnswLib.initNewIndex(index, nbItems, M, efConstruction, randomSeed);
        float[] vector1 = new float[] {1, 2, 3, 4};
        float[] vector2 = new float[] {1, 3, 2, 4};

        HnswLib.addItem(index, vector1, 1);
        HnswLib.addItem(index, vector2, 2);

        float distance = HnswLib.getDistanceBetweenLabels(index, 1, 2);
        float expectedDistance = (1.0f - 0.66666666f);
        assertRelativeError(distance, expectedDistance);
        HnswLib.destroy(index);
    }

    private void populateIndex(long index, Function<Integer, Float> getValueById, long nbItems, int dimension) {
        for (int i = 0; i < nbItems; i++) {
            float value = getValueById.apply(i);
            HnswLib.addItem(index, HnswNativeLoadTest.getVector(value, dimension), i);
            FloatByteBuf.wrappedBuffer(HnswLib.getItem(index, i));
        }
    }

    private void assertAllVectorsMatchExpected(long index, int size, Function<Integer, Float> getExpectedValue) {
        long[] ids = HnswLib.getLabels(index);
        for (int i = 0; i < ids.length; i++) {
            int id = (int)ids[i];
            float expectedValue = getExpectedValue.apply(id);
            FloatByteBuf item = FloatByteBuf.wrappedBuffer(HnswLib.getItem(index, id));
            assertAllValuesEqual(item, size, expectedValue);
        }
    }

    private void assertAllValuesEqual(FloatByteBuf vector, int size, float expectedValue) {
        assertEquals(size, vector.asFloatBuffer().limit());
        for (int j = 0; j < size; j++) {
            assertEquals(expectedValue, vector.get(j), delta);
        }
    }

    private void assertAllEqual(FloatBuffer vector1, FloatBuffer vector2, int size) {
        assertEquals(size, vector2.limit());
        assertEquals(vector1.limit(), vector2.limit());
        for (int j = 0; j < size; j++) {
            assertEquals(vector1.get(j), vector2.get(j), 1E-20);
        }
    }

    private float getL2Distance(FloatByteBuf vector1, FloatByteBuf vector2, int dimension) {
        float res = 0.0f;
        for(int i = 0; i < dimension; i++) {
            float t = vector1.get(i) - vector2.get(i);
            res += t * t;
        }
        return res;
    }
}

package com.criteo.hnsw;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class HnswNativeLoadTest {
    private int dim = 10;
    private int M = 5;
    private int efConstruction = 200;
    private int randomSeed = 42;

    @Test
    public void creating_Angular() {
        long pointer = HnswLib.create(dim, Metrics.AngularVal, Precision.Float32Val);
        HnswLib.destroy(pointer);
    }

    @Test
    public void creating_Euclidean() {
        long pointer = HnswLib.create(dim, Metrics.EuclideanVal, Precision.Float32Val);
        HnswLib.destroy(pointer);
    }

    @Test
    public void creating_InnerProduct() {
        long pointer = HnswLib.create(dim, Metrics.DotProductVal, Precision.Float32Val);
        HnswLib.destroy(pointer);
    }

    @Test
    public void getting_nb_items_returns_0_when_empty() {
        long pointer = HnswLib.create(dim, Metrics.EuclideanVal, Precision.Float32Val);
        HnswLib.initNewIndex(pointer, 100, M, efConstruction, randomSeed);
        assertEquals(0, HnswLib.getNbItems(pointer));
        HnswLib.destroy(pointer);
    }

    @Test
    public void adding_items_affects_number_of_items() {
        long pointer = HnswLib.create(dim, Metrics.EuclideanVal, Precision.Float32Val);
        int nbItems = 99;
        HnswLib.initNewIndex(pointer, nbItems, M, efConstruction, randomSeed);
        for(int i = 0; i < nbItems; i++) {
            float[] vector = getVector(1/(1f + (float)i), dim);
            HnswLib.addItem(pointer, vector, i);
        }
        assertEquals(nbItems, HnswLib.getNbItems(pointer));
        HnswLib.destroy(pointer);
    }

    static float[] getVector(float value, int dim) {
        float[] vector = new float[dim];
        for(int i = 0; i < dim; i++) {
            vector[i] = value;
        }
        return vector;
    }
}

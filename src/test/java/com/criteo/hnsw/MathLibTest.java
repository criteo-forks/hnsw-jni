package com.criteo.hnsw;

import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class MathLibTest {
    public static float delta = 1E-6f;
    public static int seed = 42;
    public static Random r = new Random(seed);

    public static int dimension = 101;
    public static FloatByteBuf zeroVector = toBuff(new float[dimension]);

    @Test
    public void l2NormIs0ForZeroVectors() {
        MathLib math = new MathLib(dimension);
        float l2norm = math.l2NormSquared(zeroVector);
        assertEquals(0.f, l2norm, delta);
    }

    @Test
    public void l2NormNonZero() {
        int nbItems = 100;
        MathLib math = new MathLib(dimension);
        for(int i = 0; i < nbItems; i++) {
            float[] vector = getRandomVector(dimension);
            float expectedL2 = 0.f;
            for (float v : vector) {
                expectedL2 += v * v;
            }
            float l2norm = math.l2NormSquared(toBuff(vector));
            float relativeError = getRelativeError(l2norm, expectedL2);
            assertTrue(relativeError < delta);
        }
    }

    @Test
    public void dotProductZero() {
        MathLib math = new MathLib(dimension);
        float dotProduct = math.innerProductDistance(zeroVector, zeroVector);
        assertEquals(0.f, dotProduct, delta);
        float[] vector = getRandomVector(dimension);
        dotProduct = math.innerProductDistance(zeroVector, toBuff(vector));
        assertEquals(0.f, dotProduct, delta);
        dotProduct = math.innerProductDistance(toBuff(vector), zeroVector);
        assertEquals(0.f, dotProduct, delta);
    }

    @Test
    public void dotProductNonZero() {
        int nbItems = 100;
        MathLib math = new MathLib(dimension);
        for(int i = 0; i < nbItems; i++) {
            float[] vector1 = getRandomVector(dimension);
            for(int j = 0; j < nbItems; j++) {
                float[] vector2 = getRandomVector(dimension);
                float expectedDotProduct = 0.f;
                for(int k = 0; k < dimension; k++) {
                    expectedDotProduct += vector1[k] * vector2[k];
                }
                float dotProduct = math.innerProductDistance(toBuff(vector1), toBuff(vector2));
                float relativeError = getRelativeError(dotProduct, expectedDotProduct);
                assertTrue(relativeError < delta);
            }
        }
    }

    public static float getRelativeError(float a, float b) {
        return  Math.max(a, b) / Math.min(a, b) - 1.f;
    }

    public static float[] getRandomVector(int dimension) {
        float[] vector = new float[dimension];
        for(int j = 0; j < dimension; j++) {
            vector[j] = r.nextFloat();
        }
        return vector;
    }

    public static FloatByteBuf toBuff(float[] xs) {
        return FloatByteBuf.wrapArray(xs);
    }
}

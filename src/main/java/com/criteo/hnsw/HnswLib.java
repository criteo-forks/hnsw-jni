package com.criteo.hnsw;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.SystemUtils;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;


public class HnswLib {
    static {
        LoadNativeLib();
    }

    public static void ensureLoaded() {
    }

    private static void LoadNativeLib() {
        try {
            if (SystemUtils.IS_OS_LINUX) {
                NativeUtils.loadLibraryFromJar("/native/libHNSWLIB_JNI.so");
            } else if (SystemUtils.IS_OS_MAC_OSX) {
                NativeUtils.loadLibraryFromJar("/native/libHNSWLIB_JNI.dylib");
            } else if (SystemUtils.IS_OS_WINDOWS) {
                NativeUtils.loadLibraryFromJar("/native/libHNSWLIB_JNI.dll");
            } else {
                throw new NotImplementedException("OS is not compatible");
            }
            System.out.println("Load HNSWLIB_JNI success");
        } catch (IOException e2) {
            throw new RuntimeException(e2);
        }
    }

    public static native long create(int dim, int distance, int precision);

    public static native void destroy(long pointer);

    public static native void initNewIndex(long pointer, long max_elements, long M, long ef_construction, long random_seed);

    public static native void initBruteforce(long pointer, long max_elements);

    public static native void enableBruteforceSearch(long pointer);

    public static native void setEf(long pointer, long ef_search);

    public static native void saveIndex(long pointer, String path);

    public static native void loadIndex(long pointer, String path);

    public static native void loadBruteforce(long pointer, String path);

    public static native void addItem(long pointer, float[] vector, long label);

    public static native void addItemBuffer(long pointer, FloatBuffer vector, long label);

    public static native long getNbItems(long pointer);

    public static native ByteBuffer getItem(long pointer, long label);

    public static native int search(long pointer, FloatBuffer query_buffer, long k, LongBuffer items_result_buffer, FloatBuffer distance_result_buffer, ByteBuffer[] result_vectors, boolean bruteforceSearch);

    public static native boolean decode(long pointer, ByteBuffer src, ByteBuffer dst);

    public static native boolean encode(long pointer, ByteBuffer src, ByteBuffer dst);

    public static native float getDistanceBetweenLabels(long index, long label1, long label2);

    public static native float getDistanceBetweenVectors(long index, FloatBuffer vector1, FloatBuffer vector2);

    public static native long[] getLabels(long index);

    public static native int getPrecision(long index);

    public static native int getMetric(long index);

    public static native int getDimension(long index);

    public static native boolean encodingNeedsTraining(long index);

    public static native void trainEncodingSpace(long pointer, float[] vector);

    public static native void trainEncodingSpaceBuffer(long pointer, FloatBuffer vector);
}

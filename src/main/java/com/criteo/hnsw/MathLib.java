package com.criteo.hnsw;

import java.nio.ByteBuffer;

public class MathLib {
    static {
        HnswLib.ensureLoaded();
    }

    private long pointer;

    public MathLib(int dimension) {
        pointer = create(dimension);
    }

    public float innerProductDistance(FloatByteBuf vector1, FloatByteBuf vector2) {
        return innerProductDistance(pointer, vector1.getNioBuffer(), vector2.getNioBuffer());
    }

    public float l2DistanceSquared(FloatByteBuf vector1, FloatByteBuf vector2) {
        return l2DistanceSquared(pointer, vector1.getNioBuffer(), vector2.getNioBuffer());
    }

    public float l2NormSquared(FloatByteBuf vector) {
        return l2NormSquared(pointer, vector.getNioBuffer());
    }

    public void destroy() {
        destroy(pointer);
    }

    public static native long create(int dimension);
    public static native float innerProductDistance(long pointer, ByteBuffer vector1, ByteBuffer vector2);
    public static native float l2DistanceSquared(long pointer, ByteBuffer vector1, ByteBuffer vector2);
    public static native float l2NormSquared(long pointer, ByteBuffer vector);
    public static native void destroy(long pointer);
}

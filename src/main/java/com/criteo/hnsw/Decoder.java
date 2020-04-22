package com.criteo.hnsw;

import java.nio.ByteBuffer;

public class Decoder {
    private long pointer;
    private int dimension;
    private String precision;

    private static final int FLOAT_16_SIZE_IN_BYTES = 2;

    private ThreadLocal<FloatByteBuf> decodedEmbedding = new ThreadLocal<FloatByteBuf>(){
        @Override protected FloatByteBuf initialValue() {
            FloatByteBuf decoded = new FloatByteBuf(dimension, false);
            decoded.writerIndex(dimension); // Since memory update happens in native, we need to tell JVM what is new position
            return decoded;
        }
    };

    public static Decoder create(int dimension, String precision) {
        int precisionVal = Precision.getVal(precision);
        // TODO: Decouple decoding from Euclidean Distance and Hnsw Index. To be done in next review
        long pointer = HnswLib.create(dimension, Metrics.EuclideanVal, precisionVal);
        return new Decoder(pointer, dimension, precision);
    }

    private Decoder(long pointer, int dimension, String precision) {
        this.pointer = pointer;
        this.dimension = dimension;
        this.precision = precision;
    }

    /**
     * Decoding of the src buffer and storing results in internal thread-local variable. To be used right before
     * computation to avoid book keeping of the memory between managed head off-heap
     * @param src
     * @return Thread-local buffer which will be overriden with the next call to decode unless it didn't require decoding
     */
    public FloatByteBuf decode(FloatByteBuf src) {
        if (precision.equals(Precision.Float16)) {
            FloatByteBuf decoded = decodedEmbedding.get();
            HnswLib.decodeFloat16(pointer, src.getNioBuffer(), decoded.getNioBuffer());
            return decoded;
        }
        return src;
    }

    public ByteBuffer encode(ByteBuffer src) {
        if (precision.equals(Precision.Float16)) {
            ByteBuffer dst = ByteBuffer.allocateDirect(dimension * FLOAT_16_SIZE_IN_BYTES);
            HnswLib.encodeFloat16(pointer, src, dst);
            return dst;
        }
        return src;
    }
}

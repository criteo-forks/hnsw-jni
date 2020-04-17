package com.criteo.hnsw;

import io.netty.buffer.ByteBuf;
import io.netty.buffer.ByteBufAllocator;
import io.netty.buffer.Unpooled;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public class FloatByteBuf implements ByteBufWrapper {
    private static final int FLOAT_SIZE = 4;
    private final ByteBuf inner;

    public FloatByteBuf(int capacity) {
        this(capacity, true);
    }

    public FloatByteBuf(int capacity, boolean pooled) {
        int bufferCapacity = capacity * FLOAT_SIZE;
        if (pooled) {
            inner = ByteBufAllocator.DEFAULT.directBuffer(bufferCapacity);
        } else {
            inner = Unpooled.directBuffer(bufferCapacity);
        }

    }

    private FloatByteBuf(ByteBuf inner) {
        this.inner = inner;
    }

    public float get(int index) {
        return this.inner.getFloatLE(index * FLOAT_SIZE);
    }

    public float read() {
        return this.inner.readFloatLE();
    }

    public FloatByteBuf writeBytes(FloatByteBuf src) {
        this.inner.writeBytes(src.inner, 0, src.inner.capacity());
        return this;
    }

    public FloatByteBuf write(float value) {
        this.inner.writeFloatLE(value);
        return this;
    }

    public static FloatByteBuf wrapArray(float[] array) {
        FloatByteBuf buf = new FloatByteBuf(array.length);
        for(float f: array) {
            buf.write(f);
        }
        return buf;
    }

    public static FloatByteBuf wrappedBuffer(ByteBuffer buffer) {
        return new FloatByteBuf(Unpooled.wrappedBuffer(buffer));
    }

    @Override
    public ByteBuffer getNioBuffer() {
        return this.inner.nioBuffer().order(ByteOrder.nativeOrder());
    }

    public FloatBuffer asFloatBuffer() {
        return this.getNioBuffer().asFloatBuffer();
    }

    public void writerIndex(int index) {
        this.inner.writerIndex(index * FLOAT_SIZE);
    }

    public FloatByteBuf writerBytes(ByteBuffer src) {
        this.inner.writeBytes(src);
        return this;
    }

    public FloatByteBuf writeZero(int length) {
        this.inner.writeZero(length * FLOAT_SIZE);
        return this;
    }

    public float[] copyToArray() {
        float[] array = new float[this.inner.readableBytes() / FLOAT_SIZE];
        for (int i = 0; i < array.length; i++) {
            array[i] = this.get(i);
        }
        return array;
    }

    public int size() {
        return this.inner.readableBytes() / FLOAT_SIZE;
    }

    @Override
    public void close() throws Exception {
        this.inner.release();
    }
}


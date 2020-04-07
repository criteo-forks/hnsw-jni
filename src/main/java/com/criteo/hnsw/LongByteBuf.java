package com.criteo.hnsw;

import io.netty.buffer.ByteBuf;
import io.netty.buffer.ByteBufAllocator;
import io.netty.buffer.Unpooled;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.LongBuffer;

public class LongByteBuf implements ByteBufWrapper {
    private static final int LONG_SIZE = 8;
    private final ByteBuf inner;

    public LongByteBuf(int capacity) {
        this(capacity, true);
    }

    public LongByteBuf(int capacity, boolean pooled) {
        int bufferCapacity = capacity * LONG_SIZE;
        if (pooled) {
            inner = ByteBufAllocator.DEFAULT.directBuffer(bufferCapacity);
        } else {
            inner = Unpooled.directBuffer(bufferCapacity);
        }
    }

    private LongByteBuf(ByteBuf inner) {
        this.inner = inner;
    }


    public long get(int index) {
        return this.inner.getLongLE(index * LONG_SIZE);
    }

    public long read() {
        return this.inner.readLongLE();
    }

    public LongByteBuf write(long value) {
        this.inner.writeLongLE(value);
        return this;
    }

    public LongByteBuf writeBytes(LongByteBuf src) {
        this.inner.writeBytes(src.inner, 0, src.inner.capacity());
        return this;
    }

    public static LongByteBuf wrapArray(long[] array) {
        LongByteBuf buf = new LongByteBuf(array.length);
        for(long f: array) {
            buf.write(f);
        }
        return buf;
    }

    public static LongByteBuf wrappedBuffer(ByteBuffer buffer) {
        return new LongByteBuf(Unpooled.wrappedBuffer(buffer));
    }

    @Override
    public ByteBuffer getNioBuffer() {
        return this.inner.nioBuffer().order(ByteOrder.nativeOrder());
    }
    public LongBuffer asLongBuffer() {
        return this.getNioBuffer().asLongBuffer();
    }

    public void writerIndex(int index) {
        this.inner.writerIndex(index * LONG_SIZE);
    }

    public LongByteBuf writeZero(int length) {
        this.inner.writeZero(length * LONG_SIZE);
        return this;
    }

    public long[] copyToArray() {
        long[] array = new long[this.inner.readableBytes() / LONG_SIZE];
        for (int i = 0; i < array.length; i++) {
            array[i] = this.get(i);
        }
        return array;
    }

    @Override
    public void close() throws Exception {
        this.inner.release();
    }
}
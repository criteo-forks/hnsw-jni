package com.criteo.hnsw;

import io.netty.buffer.ByteBuf;
import io.netty.buffer.ByteBufAllocator;
import io.netty.buffer.Unpooled;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.LongBuffer;

public final class LongByteBuf implements ByteBufWrapper {

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

  public final long get(int index) {
    return inner.getLongLE(index * LONG_SIZE);
  }

  public final long read() {
    return inner.readLongLE();
  }

  public final LongByteBuf writeBytes(LongByteBuf src) {
    inner.writeBytes(src.inner, 0, src.inner.capacity());
    return this;
  }

  public final LongByteBuf writerBytes(ByteBuffer src) {
    inner.writeBytes(src);
    return this;
  }

  public final LongByteBuf write(long value) {
    inner.writeLongLE(value);
    return this;
  }

  public static LongByteBuf wrapArray(long[] array) {
    final LongByteBuf buf = new LongByteBuf(array.length);
    for (long l : array) {
      buf.write(l);
    }
    return buf;
  }

  public static LongByteBuf wrappedBuffer(ByteBuffer buffer) {
    return new LongByteBuf(Unpooled.wrappedBuffer(buffer));
  }

  @Override
  public final ByteBuffer getNioBuffer() {
    return inner.nioBuffer().order(ByteOrder.nativeOrder());
  }

  public final LongBuffer asLongBuffer() {
    return getNioBuffer().asLongBuffer();
  }

  public final void readerIndex(int index) {
    inner.readerIndex(index * LONG_SIZE);
  }

  public final void writerIndex(int index) {
    inner.writerIndex(index * LONG_SIZE);
  }

  public final LongByteBuf writeZero(int length) {
    inner.writeZero(length * LONG_SIZE);
    return this;
  }

  public final long[] copyToArray() {
    final long[] array = new long[size()];
    for (int i = 0; i < array.length; i++) {
      array[i] = get(i);
    }
    return array;
  }

  public final int size() {
    return inner.readableBytes() / LONG_SIZE;
  }

  @Override
  public final void close() throws Exception {
    inner.release();
  }
}

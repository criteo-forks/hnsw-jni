package com.criteo.hnsw;

import io.netty.buffer.ByteBuf;
import io.netty.buffer.ByteBufAllocator;
import io.netty.buffer.Unpooled;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public final class FloatByteBuf implements ByteBufWrapper {

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

  public final float get(int index) {
    return inner.getFloatLE(index * FLOAT_SIZE);
  }

  public final float read() {
    return inner.readFloatLE();
  }

  public final FloatByteBuf writeBytes(FloatByteBuf src) {
    inner.writeBytes(src.inner, 0, src.inner.capacity());
    return this;
  }

  public final FloatByteBuf writerBytes(ByteBuffer src) {
    inner.writeBytes(src);
    return this;
  }

  public final FloatByteBuf write(float value) {
    inner.writeFloatLE(value);
    return this;
  }

  public static FloatByteBuf wrapArray(float[] array) {
    final FloatByteBuf buf = new FloatByteBuf(array.length);
    for (float f : array) {
      buf.write(f);
    }
    return buf;
  }

  public static FloatByteBuf wrappedBuffer(ByteBuffer buffer) {
    return new FloatByteBuf(Unpooled.wrappedBuffer(buffer));
  }

  @Override
  public final ByteBuffer getNioBuffer() {
    return inner.nioBuffer().order(ByteOrder.nativeOrder());
  }

  public final FloatBuffer asFloatBuffer() {
    return getNioBuffer().asFloatBuffer();
  }

  public final void readerIndex(int index) {
    inner.readerIndex(index * FLOAT_SIZE);
  }

  public final void writerIndex(int index) {
    inner.writerIndex(index * FLOAT_SIZE);
  }

  public final FloatByteBuf writeZero(int length) {
    inner.writeZero(length * FLOAT_SIZE);
    return this;
  }

  public final float[] copyToArray() {
    final float[] array = new float[size()];
    for (int i = 0; i < array.length; i++) {
      array[i] = get(i);
    }
    return array;
  }

  public final int size() {
    return inner.readableBytes() / FLOAT_SIZE;
  }

  @Override
  public final void close() throws Exception {
    inner.release();
  }
}

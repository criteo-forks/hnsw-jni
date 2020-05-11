package com.criteo.hnsw;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.Test;

public class FloatByteBufTest {

  @Test
  public void testReadWrite() {
    FloatByteBuf floatByteBuf = new FloatByteBuf(3, false);
    assertEquals(0, floatByteBuf.size());

    floatByteBuf.write(7.7f);
    assertEquals(1, floatByteBuf.size());
    assertEquals(7.7f, floatByteBuf.get(0), 1E-6f);
    assertEquals(7.7f, floatByteBuf.read(), 1E-6f);

    floatByteBuf.write(1.11f);
    assertEquals(7.7f, floatByteBuf.get(0), 1E-6f);
    assertEquals(1.11f, floatByteBuf.get(1), 1E-6f);
    assertEquals(1.11f, floatByteBuf.read(), 1E-6f);

    floatByteBuf.write(3.33f);
    assertEquals(7.7f, floatByteBuf.get(0), 1E-6f);
    assertEquals(1.11f, floatByteBuf.get(1), 1E-6f);
    assertEquals(3.33f, floatByteBuf.get(2), 1E-6f);
    assertEquals(3.33f, floatByteBuf.read(), 1E-6f);

    floatByteBuf.writeZero(4);

    floatByteBuf.write(5.555f);
    assertEquals(5.555f, floatByteBuf.get(7), 1E-6f);
    // Here, we read the latest not read value which is `0'.
    assertEquals(0f, floatByteBuf.read(), 1E-6f);

    floatByteBuf.readerIndex(0);
    floatByteBuf.writerIndex(8);
    assertEquals(8, floatByteBuf.size());

    float[] expectedArray = new float[]{7.7f, 1.11f, 3.33f, 0f, 0f, 0f, 0f, 5.555f};

    assertArrayEquals(expectedArray, floatByteBuf.copyToArray(), 1E-6f);

    ByteBuffer byteBuffer = floatByteBuf.getNioBuffer();
    float[] actualArray = new float[expectedArray.length];
    byteBuffer.asFloatBuffer().get(actualArray);
    assertArrayEquals(expectedArray, actualArray, 1E-6f);
  }

  @Test
  public void testConsumeBuffer() {
    float[] expectedArray = new float[]{1f, 2.2f, 3.33f, 4.444f, 5.5555f, 6.66666f, 7.777777f};

    ByteBuffer tmpByteBuffer = ByteBuffer.allocate(expectedArray.length * 4).order(ByteOrder.nativeOrder());
    tmpByteBuffer
        .putFloat(expectedArray[0])
        .putFloat(expectedArray[1])
        .putFloat(expectedArray[2])
        .putFloat(expectedArray[3])
        .putFloat(expectedArray[4])
        .putFloat(expectedArray[5])
        .putFloat(expectedArray[6]);
    byte[] rawBytes1 = tmpByteBuffer.array();
    FloatByteBuf floatByteBuf1 = new FloatByteBuf(expectedArray.length, false);
    floatByteBuf1.writerBytes(ByteBuffer.wrap(rawBytes1));
    floatByteBuf1.writerIndex(expectedArray.length);
    assertArrayEquals(expectedArray, floatByteBuf1.copyToArray(), 1E-6f);

    FloatByteBuf tmpFloatBuffer = FloatByteBuf.wrapArray(expectedArray);
    byte[] rawBytes2 = new byte[expectedArray.length * 4];
    tmpFloatBuffer.getNioBuffer().get(rawBytes2);
    FloatByteBuf floatByteBuf2 = new FloatByteBuf(expectedArray.length, false);
    floatByteBuf2.writeBytes(tmpFloatBuffer);
    floatByteBuf2.writerIndex(expectedArray.length);
    assertArrayEquals(expectedArray, floatByteBuf2.copyToArray(), 1E-6f);

    assertArrayEquals(rawBytes1, rawBytes2);
  }
}

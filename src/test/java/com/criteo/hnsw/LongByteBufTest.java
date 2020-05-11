package com.criteo.hnsw;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.Test;

public class LongByteBufTest {

  @Test
  public void testReadWrite() {
    LongByteBuf longByteBuf = new LongByteBuf(3, false);
    assertEquals(0, longByteBuf.size());

    longByteBuf.write(-333L);
    assertEquals(1, longByteBuf.size());
    assertEquals(-333L, longByteBuf.get(0));
    assertEquals(-333L, longByteBuf.read());

    longByteBuf.write(-22L);
    assertEquals(-333L, longByteBuf.get(0));
    assertEquals(-22L, longByteBuf.get(1));
    assertEquals(-22L, longByteBuf.read());

    longByteBuf.write(-1L);
    assertEquals(-333L, longByteBuf.get(0));
    assertEquals(-22L, longByteBuf.get(1));
    assertEquals(-1L, longByteBuf.get(2));
    assertEquals(-1L, longByteBuf.read());

    longByteBuf.writeZero(4);

    longByteBuf.write(7777777L);
    assertEquals(7777777L, longByteBuf.get(7));
    // Here, we read the latest not read value which is `0'.
    assertEquals(0L, longByteBuf.read());

    longByteBuf.readerIndex(0);
    longByteBuf.writerIndex(8);
    assertEquals(8, longByteBuf.size());

    long[] expectedArray = new long[]{-333L, -22L, -1L, 0L, 0L, 0L, 0L, 7777777L};

    assertArrayEquals(expectedArray, longByteBuf.copyToArray());

    ByteBuffer byteBuffer = longByteBuf.getNioBuffer();
    long[] actualArray = new long[expectedArray.length];
    byteBuffer.asLongBuffer().get(actualArray);
    assertArrayEquals(expectedArray, actualArray);
  }

  @Test
  public void testConsumeBuffer() {
    long[] expectedArray = new long[]{-333L, -22L, -1L, 0L, 1L, 22L, 333L};

    ByteBuffer tmpByteBuffer = ByteBuffer.allocate(expectedArray.length * 8).order(ByteOrder.nativeOrder());
    tmpByteBuffer
        .putLong(expectedArray[0])
        .putLong(expectedArray[1])
        .putLong(expectedArray[2])
        .putLong(expectedArray[3])
        .putLong(expectedArray[4])
        .putLong(expectedArray[5])
        .putLong(expectedArray[6]);
    byte[] rawBytes1 = tmpByteBuffer.array();
    LongByteBuf longByteBuf1 = new LongByteBuf(expectedArray.length, false);
    longByteBuf1.writerBytes(ByteBuffer.wrap(rawBytes1));
    longByteBuf1.writerIndex(expectedArray.length);
    assertArrayEquals(expectedArray, longByteBuf1.copyToArray());

    LongByteBuf tmpLongBuffer = LongByteBuf.wrapArray(expectedArray);
    byte[] rawBytes2 = new byte[expectedArray.length * 8];
    tmpLongBuffer.getNioBuffer().get(rawBytes2);
    LongByteBuf longByteBuf2 = new LongByteBuf(expectedArray.length, false);
    longByteBuf2.writeBytes(tmpLongBuffer);
    longByteBuf2.writerIndex(expectedArray.length);
    assertArrayEquals(expectedArray, longByteBuf2.copyToArray());

    assertArrayEquals(rawBytes1, rawBytes2);
  }
}

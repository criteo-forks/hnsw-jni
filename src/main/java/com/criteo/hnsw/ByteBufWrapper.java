package com.criteo.hnsw;

import java.nio.ByteBuffer;

public interface ByteBufWrapper extends AutoCloseable {
    ByteBuffer getNioBuffer();
}

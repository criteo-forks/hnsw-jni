package com.criteo.hnsw;

public class Precision {
    public static final String Float32 = "float32";
    public static final String Float16 = "float16";

    // See mapping int hnswindex.h `Precision` enum
    public static final int Float32Val = 1;
    public static final int Float16Val = 2;

    public static int getVal(String str) {
        switch (str) {
            case Precision.Float32: return Float32Val;
            case Precision.Float16: return Float16Val;
            default: throw new UnsupportedOperationException();
        }
    }
}

package com.criteo.hnsw;

public class Precision {
    public static final String Float32 = "float32";
    public static final String Float16 = "float16";
    public static final String Float8 = "float8";

    // See mapping int hnswindex.h `Precision` enum
    public static final int Float32Val = 1;
    public static final int Float16Val = 2;
    public static final int Float8Val = 3;

    public static final int FLOAT_32_SIZE_IN_BYTES = 4;
    public static final int FLOAT_16_SIZE_IN_BYTES = 2;
    public static final int FLOAT_8_SIZE_IN_BYTES = 1;

    public static int getVal(String str) {
        switch (str) {
            case Precision.Float32: return Float32Val;
            case Precision.Float16: return Float16Val;
            case Precision.Float8: return Float8Val;
            default: throw new UnsupportedOperationException();
        }
    }

    public static String getStr(int val) {
        switch (val) {
            case Precision.Float32Val: return Float32;
            case Precision.Float16Val: return Float16;
            case Precision.Float8Val: return Float8;
            default: throw new UnsupportedOperationException();
        }
    }

    public static int getBytesPerComponent(int precision) {
        switch (precision) {
            case Precision.Float32Val: return FLOAT_32_SIZE_IN_BYTES;
            case Precision.Float16Val: return FLOAT_16_SIZE_IN_BYTES;
            case Precision.Float8Val: return FLOAT_8_SIZE_IN_BYTES;
            default: throw new UnsupportedOperationException();
        }
    }

    public static int getBytesPerComponent(String precision) {
        switch (precision) {
            case Precision.Float32: return FLOAT_32_SIZE_IN_BYTES;
            case Precision.Float16: return FLOAT_16_SIZE_IN_BYTES;
            case Precision.Float8: return FLOAT_8_SIZE_IN_BYTES;
            default: throw new UnsupportedOperationException();
        }
    }
}

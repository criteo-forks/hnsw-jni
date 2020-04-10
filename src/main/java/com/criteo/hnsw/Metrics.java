package com.criteo.hnsw;

public class Metrics {
    public static final String Euclidean = "Euclidean";
    public static final String Angular = "Angular";
    public static final String DotProduct = "DotProduct";
    public static final String Kendall = "Kendall";

    // See mapping int hnswindex.h `Distance` enum
    public static final int EuclideanVal = 1;
    public static final int AngularVal = 2;
    public static final int DotProductVal = 3;
    public static final int KendallVal = 4;

    public static int getVal(String str) {
        switch (str) {
            case Metrics.Euclidean: return EuclideanVal;
            case Metrics.Angular: return AngularVal;
            case Metrics.DotProduct: return  DotProductVal;
            case Metrics.Kendall: return KendallVal;
            default: throw new UnsupportedOperationException();
        }
    }
}

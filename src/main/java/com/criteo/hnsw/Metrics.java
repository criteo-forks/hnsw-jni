package com.criteo.hnsw;

import com.criteo.knn.knninterface.Metric;

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
            case Metrics.Euclidean:
                return EuclideanVal;
            case Metrics.Angular:
                return AngularVal;
            case Metrics.DotProduct:
                return DotProductVal;
            case Metrics.Kendall:
                return KendallVal;
            default:
                throw new UnsupportedOperationException();
        }
    }

    public static Metric getEnumVal(String str) {
        switch (str) {
            case Metrics.Euclidean:
                return Metric.Euclidean;
            case Metrics.Angular:
                return Metric.Cosine;
            case Metrics.DotProduct:
                return Metric.InnerProduct;
            case Metrics.Kendall:
                return Metric.Kendall;
            default:
                throw new UnsupportedOperationException();
        }
    }

    public static String getStringVal(Metric metric) {
        switch (metric) {
            case Euclidean:
                return Metrics.Euclidean;
            case Cosine:
                return Metrics.Angular;
            case InnerProduct:
                return Metrics.DotProduct;
            case Kendall:
                return Metrics.Kendall;
            default:
                throw new UnsupportedOperationException();
        }
    }
}

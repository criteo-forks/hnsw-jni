package com.criteo.hnsw;

import java.nio.ByteBuffer;

public class HnswIndex {
    private long pointer;
    private int dimension;


    public static HnswIndex create(String metric, int dimension) {
        long pointer;
        switch (metric) {
            case Metrics.Angular: pointer = HnswLib.createAngular(dimension); break;
            case Metrics.Euclidean: pointer = HnswLib.createEuclidean(dimension); break;
            case Metrics.DotProduct: pointer = HnswLib.createInnerProduct(dimension); break;
            case Metrics.Kendall: pointer = HnswLib.createKendall(dimension); break;
            default: throw new UnsupportedOperationException();
        }

        return new HnswIndex(pointer, dimension);
    }

    public static HnswIndex createF16(String metric, int dimension) {
        long pointer;
        switch (metric) {
            case Metrics.Euclidean: pointer = HnswLib.createEuclideanF16(dimension); break;
            default: throw new UnsupportedOperationException();
        }

        return new HnswIndex(pointer, dimension);
    }

    public HnswIndex(long pointer, int dimension) {
        this.pointer = pointer;
        this.dimension = dimension;
    }

    public long getPointer() {
        return pointer;
    }

    public long getDimension() {
        return dimension;
    }

    public long load(String path) {
        HnswLib.loadIndex(pointer, path);
        return getNbItems();
    }

    public void addItem(float[] vector, long id) {
        HnswLib.addItem(pointer, vector, id);
    }

    public void save(String path) {
        HnswLib.saveIndex(pointer, path);
    }

    public void initNewIndex(long maxElements, long M, long efConstruction, long randomSeed) {
        HnswLib.initNewIndex(pointer, maxElements, M, efConstruction, randomSeed);
    }

    public void setEf(long ef) {
        HnswLib.setEf(pointer, ef);
    }

    public void unload() {
        HnswLib.destroy(pointer);
    }

    public long getNbItems() {
        return HnswLib.getNbItems(pointer);
    }

    public long[] getIds() {
        return HnswLib.getLabels(this.pointer);
    }

    public FloatByteBuf getItem(long label) {
        ByteBuffer buffer = HnswLib.getItem(this.pointer, label);

        if (buffer != null) {
            return FloatByteBuf.wrappedBuffer(buffer);
        } else {
            return null;
        }
    }

    public KnnResult search(FloatByteBuf query, int k) throws Exception {
        try(LongByteBuf result_item = new LongByteBuf(k)) {
            try(FloatByteBuf result_distance = new FloatByteBuf(k)) {
                ByteBuffer[] result_vectors = new ByteBuffer[k];
                int resultCount = HnswLib.search(pointer, query.asFloatBuffer(), k,
                        result_item.asLongBuffer(),
                        result_distance.asFloatBuffer(),
                        result_vectors
                );
                result_item.writerIndex(resultCount);
                result_distance.writerIndex(resultCount);

                KnnResult result = new KnnResult();
                result.resultCount = resultCount;
                result.resultDistances = new float[resultCount];
                result.resultItems = new long[resultCount];
                result.resultVectors = new FloatByteBuf[resultCount];

                for (int i = 0; i < resultCount; i++) {
                    result.resultDistances[i] = result_distance.read();
                    result.resultItems[i] = result_item.read();
                    result.resultVectors[i] = FloatByteBuf.wrappedBuffer(result_vectors[i]);
                }

                return result;
            }
        }
    }

    public float getDistance(FloatByteBuf vector1, FloatByteBuf vector2) {
        return HnswLib.getDistanceBetweenVectors(pointer, vector1.asFloatBuffer(), vector2.asFloatBuffer());
    }

}

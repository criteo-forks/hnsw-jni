package com.criteo.hnsw;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

public class HnswIndex {
    private long pointer;
    private int dimension;
    private int precision;
    private Boolean isBruteforce;

    public static HnswIndex create(String metric, int dimension) {
        return create(metric, dimension, Precision.Float32);
    }

    public static HnswIndex create(String metric, int dimension, String precision) {
        return create(metric, dimension, precision, false);
    }

    public static HnswIndex create(String metric, int dimension, String precision, Boolean isBruteforce) {
        int metricVal = Metrics.getVal(metric);
        int precisionVal = Precision.getVal(precision);
        return create(metricVal, dimension, precisionVal, isBruteforce);
    }

    public static HnswIndex create(int metricVal, int dimension, int precisionVal, Boolean isBruteforce) {
        long pointer = HnswLib.create(dimension, metricVal, precisionVal);
        return new HnswIndex(pointer, isBruteforce);
    }

    public HnswIndex(long pointer, Boolean isBruteforce) {
        this.pointer = pointer;
        this.isBruteforce = isBruteforce;
        this.precision = HnswLib.getPrecision(pointer);
        this.dimension = HnswLib.getDimension(pointer);
    }

    public long getPointer() {
        return pointer;
    }

    public int getMetric() {
        return HnswLib.getMetric(pointer);
    }

    public Boolean isBruteforce() {
        return isBruteforce;
    }

    public int getDimension() {
        return HnswLib.getDimension(pointer);
    }

    public int getPrecision() {
        return HnswLib.getPrecision(pointer);
    }

    public long load(String path) {
        if (isBruteforce) {
            HnswLib.loadBruteforce(pointer, path);
        } else {
            HnswLib.loadIndex(pointer, path);
        }
        return getNbItems();
    }

    public void addItem(float[] vector, long id) {
        HnswLib.addItem(pointer, vector, id);
    }

    public void addItem(FloatBuffer vector, long id) {
        HnswLib.addItemBuffer(pointer, vector, id);
    }

    public void save(String path) {
        HnswLib.saveIndex(pointer, path);
    }

    public void initNewIndex(long maxElements, long M, long efConstruction, long randomSeed) {
        HnswLib.initNewIndex(pointer, maxElements, M, efConstruction, randomSeed);
    }

    public void initBruteforce(long maxElements) {
        HnswLib.initBruteforce(pointer, maxElements);
    }

    public void enableBruteforceSearch() {
        HnswLib.enableBruteforceSearch(pointer);
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
        if(buffer == null) {
            return null;
        }
        return FloatByteBuf.wrappedBuffer(buffer);
    }

    public FloatByteBuf getItemDecoded(long label) throws Exception{
        try(FloatByteBuf item = getItem(label)) {
            return decode(item);
        }
    }

    public KnnResult search(FloatByteBuf query, int k) throws Exception {
        return search(query, k, false);
    }

    public KnnResult searchBruteforce(FloatByteBuf query, int k) throws Exception {
        return search(query, k, true);
    }

    public KnnResult search(FloatByteBuf query, int k, boolean bruteforceSearch) throws Exception {
        try(LongByteBuf result_item = new LongByteBuf(k)) {
            try(FloatByteBuf result_distance = new FloatByteBuf(k)) {
                ByteBuffer[] result_vectors = new ByteBuffer[k];
                int resultCount = HnswLib.search(pointer, query.asFloatBuffer(), k,
                        result_item.asLongBuffer(),
                        result_distance.asFloatBuffer(),
                        result_vectors,
                        bruteforceSearch
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

    public float getDistanceBetweenLabels(long label1, long label2) {
        return HnswLib.getDistanceBetweenLabels(pointer, label1, label2);
    }

    public FloatByteBuf decode(FloatByteBuf src) {
        if (src == null) {
            return null;
        }
        if (precision == Precision.Float32Val) {
            FloatByteBuf ret = new FloatByteBuf(dimension);
            ret.writeBytes(src);
            return ret;
        }
        FloatByteBuf decoded = new FloatByteBuf(dimension);
        HnswLib.decode(pointer, src.getNioBuffer(), decoded.getNioBuffer());
        decoded.writerIndex(dimension); // Since memory update happens in native, we need to tell JVM what is new position
        return decoded;
    }

    public ByteBuffer encode(ByteBuffer src) {
        if (precision == Precision.Float32Val) {
            return src;
        }
        int size_per_component = Precision.getBytesPerComponent(precision);
        ByteBuffer dst = ByteBuffer.allocateDirect(dimension * size_per_component);
        HnswLib.encode(pointer, src, dst);
        return dst;
    }

    public boolean needsTraining() {
        return HnswLib.encodingNeedsTraining(pointer);
    }

    public void trainEncodingSpace(float[] vector) {
        HnswLib.trainEncodingSpace(pointer, vector);
    }

    public void trainEncodingSpace(FloatBuffer vector) {
        HnswLib.trainEncodingSpaceBuffer(pointer, vector);
    }

}

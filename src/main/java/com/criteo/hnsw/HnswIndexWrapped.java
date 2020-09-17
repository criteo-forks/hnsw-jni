package com.criteo.hnsw;

import com.criteo.knn.knninterface.FloatByteBuf;
import com.criteo.knn.knninterface.Index;
import com.criteo.knn.knninterface.KnnResult;
import com.criteo.knn.knninterface.Metric;

import java.util.Map;


public class HnswIndexWrapped implements Index {

    HnswIndex hnswIndex;

    public HnswIndexWrapped(HnswIndex hnswIndex) {
        this.hnswIndex = hnswIndex;
    }

    public static HnswIndexWrapped create(Metric metric, int dimension, Map<String, String> params) {

        String metricString = Metrics.getStringVal(metric);

        HnswIndex hnswIndex;

        if (params.containsKey("precision")) {
            if (params.containsKey("isBruteforce")) {
                hnswIndex = HnswIndex.create(metricString, dimension, params.get("precision"), Boolean.parseBoolean(params.get("isBruteforce")));
            } else {
                hnswIndex = HnswIndex.create(metricString, dimension, params.get("precision"));
            }
        } else {
            hnswIndex = HnswIndex.create(Metrics.getStringVal(metric), dimension);
        }

        if (params.containsKey("isBruteforce") && Boolean.parseBoolean(params.get("isBruteforce"))) {
            hnswIndex.initBruteforce(Long.parseLong(params.getOrDefault("maxElements", "0")));
        } else {
            long M = Long.parseLong(params.get("M"));
            long maxElements = Long.parseLong(params.get("maxElements"));
            long efConstruction = Long.parseLong(params.get("efConstruction"));
            long randomSeed = Long.parseLong(params.get("randomSeed"));
            long efSearch = Long.parseLong(params.get("efSearch"));
            hnswIndex.initNewIndex(maxElements, M, efConstruction, randomSeed);
            hnswIndex.setEf(efSearch);
        }

        return new HnswIndexWrapped(hnswIndex);
    }

    public static HnswIndexWrapped loadIndex(Metric metric, int dimension, String path, Map<String, String> params) {
        String metricString = Metrics.getStringVal(metric);
        HnswIndexWrapped index = new HnswIndexWrapped(HnswIndex.create(metricString, dimension, params.getOrDefault("precision", "float32")));
        index.readIndex(path);
        index.setHyperParameters(params);
        return index;
    }

    private void setHyperParameters(Map<String, String> params) {
        if (params.containsKey("efSearch")) {
            long efSearch = Integer.parseInt(params.get("efSearch"));
            hnswIndex.setEf(efSearch);
        }
    }

    /**
     * Internal training with a subset of the dataset, required for some types of indexes.
     *
     * @param newVectors array of vectors necessary for the training.
     */
    @Override
    public void train(float[][] newVectors) {
        for (float[] newVector : newVectors) {
            hnswIndex.trainEncodingSpace(newVector);
        }
    }

    /**
     * Return true if the index needs to be trained before using it.
     *
     * @return true if the index needs to be trained.
     */
    @Override
    public boolean needsTraining() {
        return hnswIndex.needsTraining();
    }

    /**
     * Add an array of vectors to the Knn index.
     *
     * @param vectors array of vectors to add to the index.
     */
    @Override
    public void addItems(float[][] vectors, long[] ids) {
        assert vectors.length == ids.length;
        for (int i = 0; i < vectors.length; i++) {
            addItem(vectors[i], ids[i]);
        }
    }

    /**
     * Add a vector the the Knn index.
     *
     * @param vector vector to add to the index.
     */
    @Override
    public void addItem(float[] vector, long id) {
        hnswIndex.addItem(vector, id);
    }

    /**
     * Write the index in a file in order to retrieve it later.
     *
     * @param path path on disk of the file where to write the index.
     */
    @Override
    public void writeIndex(String path) {
        hnswIndex.save(path);
    }

    /**
     * Get the number of elements in the Knn index.
     *
     * @return number of vectors in the Knn index.
     */
    @Override
    public int getNbItems() {
        return (int) hnswIndex.getNbItems();
    }

    /**
     * Get the list of ids of the vectors in the index.
     *
     * @return list of ids.
     */
    @Override
    public long[] getIds() {
        return hnswIndex.getIds();
    }

    /**
     * Return the vector of id i (or an approximation depending of the index).
     *
     * @param i id of the vector to retrieve.
     * @return Reconstructed vector of id i.
     */
    @Override
    public com.criteo.knn.knninterface.FloatByteBuf reconstruct(long i) throws Exception {

        return hnswIndex.getItemDecoded(i);
    }

    private KnnResult decodeResultVectors(KnnResult knnResult) throws Exception {

        if (hnswIndex.getPrecision() != Precision.Float32Val) {
            // unavoidable copy here when casting to float32
            for (int i = 0; i < knnResult.resultVectors.length; i++) {
                FloatByteBuf tmpBuf = hnswIndex.decode(knnResult.resultVectors[i]);
                knnResult.resultVectors[i].close();
                knnResult.resultVectors[i] = tmpBuf;
            }
        }
        return knnResult;
    }

    /**
     * Return k nearest neighbours of the query vector.
     *
     * @param query query vector for the similarity search.
     * @param k     number of neighbours to retrieve.
     * @return KnnResult object containing the distances, ids and vectors near the query vector.
     */
    @Override
    public KnnResult search(FloatByteBuf query, int k) throws Exception {

        return decodeResultVectors(hnswIndex.search(query, k));
    }

    /**
     * Read a previously saved index file and load it by overwriting the current index.
     *
     * @param path path of the file containing the saved index.
     */
    @Override
    public void readIndex(String path) {
        hnswIndex.load(path);
    }

    /**
     * Deallocate the Ram used for the Knn index.
     */
    @Override
    public void destroy() {
        hnswIndex.unload();
    }
}

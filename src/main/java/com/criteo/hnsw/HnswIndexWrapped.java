package com.criteo.hnsw;

import com.criteo.knn.knninterface.FloatByteBuf;
import com.criteo.knn.knninterface.Index;
import com.criteo.knn.knninterface.KnnResult;
import com.criteo.knn.knninterface.Metric;
import com.google.gson.Gson;

import java.util.*;


public class HnswIndexWrapped implements Index {

    private final HnswIndex hnswIndex;
    private final Map<String, String> indexParams;

    public HnswIndexWrapped(HnswIndex hnswIndex, Map<String, String> indexParams) {
        this.hnswIndex = hnswIndex;
        this.indexParams = new HashMap<>(indexParams);
    }

    /**
     * Function that creates an HnswIndexWrapped index.
     * <p>
     * All the possible parameter keys taken in input in the Map<String, String> params variable are:
     * - precision
     * - isBruteforce
     * - maxElements
     * - randomSeed
     * - efSearch
     * - efConstruction
     * - M
     * <p>
     * -> The default value of the "precision" field is "float32" (if required).
     * -> The default value of the "isBruteforce" field is "false" (if required).
     * <p>
     * If "isBruteforce" is set to "true", the following parameters are requiered:
     * -> [M, maxElements, efConstruction, efSearch, randomSeed]
     * If "isBruteforce" is set to "false", the following parameters are requiered:
     * -> [maxElements] (maxElements default value is "0")
     *
     * @param metric    distance between queries and vectors (inner product, L2, ...).
     * @param dimension dimension of the vectors in the database.
     * @param params    dictionary of parameters necessary for the loading of the index, see readme for more info
     * @return The created HnswIndexWrapped object implementing the Index interface.
     */
    public static HnswIndexWrapped create(Metric metric, int dimension, Map<String, String> params) {

        String metricString = Metrics.getStringVal(metric);

        String precision = params.getOrDefault("precision", "float32");
        boolean isBruteforce = Boolean.parseBoolean(params.getOrDefault("isBruteforce", "false"));

        HnswIndex hnswIndex;

        hnswIndex = HnswIndex.create(metricString, dimension, precision, isBruteforce);

        if (isBruteforce) {
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

        return new HnswIndexWrapped(hnswIndex, params);
    }

    /**
     * Function that loads an HnswIndexWrapped index.
     * <p>
     * In order to load the index you always need to specify some keys in the params Map:
     * - precision -> float precision, default to "float32"
     * - efSearch -> hnsw hyper-parameter, it's unclear whether this parameter is saved or not in the file containing
     * the index since it is redefined all the time. It's better to set a value here.
     *
     * @param metric    distance between queries and vectors (inner product, L2, ...).
     * @param dimension dimension of the vectors in the database.
     * @param path      path to the saved model.
     * @param params    dictionary of parameters necessary for the loading of the index, see readme for more info.
     * @return The loaded HnswIndexWrapped object implementing the Index interface.
     */
    public static HnswIndexWrapped loadIndex(Metric metric, int dimension, String path, Map<String, String> params) {
        String metricString = Metrics.getStringVal(metric);
        if (!params.containsKey("precision")) {
            params.put("precision", "float32");
        }
        HnswIndexWrapped index = new HnswIndexWrapped(HnswIndex.create(metricString, dimension, params.get("precision")), params);
        index.readIndex(path);
        index.setHyperParameters(params);
        return index;
    }

    /**
     * Set index hyper-parameters.
     *
     * @param params dictionary of hyper-parameters
     *               <p>
     *               params should only contain the key "efSearch" if available and relevant, all the other keys are ignored.
     */
    private void setHyperParameters(Map<String, String> params) {
        if (params.containsKey("efSearch")) {
            this.indexParams.replace("efSearch", params.get("efSearch"));
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
     * Gives the maximal reconstruction error of the vectors of the index
     */
    public float getExpectedReconstructionError() {
        switch (hnswIndex.getPrecision()) {
            case Precision.Float32Val:
                return 3E-7f;
            case Precision.Float16Val:
                return 2E-4f;
            default:
                return 1E7f;
        }
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
     * Get the construction parameters and hyper-parameters of the index.
     *
     * @return a json string describing the parameters.
     */
    String getIndexParameters() {
        return new Gson().toJson(this.indexParams);
    }

    /**
     * Deallocate the Ram used for the Knn index.
     */
    @Override
    public void destroy() {
        hnswIndex.unload();
    }
}

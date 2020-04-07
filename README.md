# hnswlib low-overhead JVM wrapper
JNI bindings for [HNSW kNN library](https://github.com/nmslib/hnswlib) 
The intent is provide minimum overhead implmentation for JVM.

## Building
```
gradle wrapper --gradle-version=4.10
./gradlew build
```
Will generate JNI bindings in Java and C and compile them for the current platform.


## Usage
Here is example that creates 15k embeddings (size 100), saving them into the index on disk, reading the index and retrieving some distances.

```java

package com.criteo.hnsw;

import java.io.File;
import java.util.Random;

public class HelloHnsw {
    public static Random r = new Random();
    public static void main(String[] args) throws Exception {
        int dimension = 100;
        int nbItems = 15000;
        int M = 16;
        int efConstruction = 200;

        HnswIndex index = HnswIndex.create(Metrics.Euclidean, dimension);

        index.initNewIndex(nbItems, M, efConstruction, 42);
        for (int i = 0; i < nbItems; i++) {
            float[] vector = getRandomVector(dimension);
            index.addItem(vector, i);
        }

        String path = new File("./bin", "index.hnsw").toString();

        index.save(path);

        index.load(path);
        System.out.println("Loaded" + index.getNbItems() + " items");
        long[] ids = index.getIds();

        for (int i = 0; i < 20; i++) {
            long queryId = ids[i];
            FloatByteBuf query = index.getItem(queryId);
            KnnResult result = index.search(query, 3);
            for (int j = 0; j < result.resultCount; j++) {
                System.out.println(queryId + " -> " + result.resultItems[i] + " distance: " + result.resultDistances[i]);
            }
        }
        index.unload();
    }

    public static float[] getRandomVector(int dimension) {
        float[] vector = new float[dimension];
        for(int i = 0; i < dimension; i++) {
            vector[i] = r.nextFloat();
        }
        return vector;
    }
}

```

To run same example from repo:

```
./gradlew run
```

To run tests:

```
./gradlew test
```
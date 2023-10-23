package globalquake.client;

import globalquake.core.earthquake.ClusterAnalysis;
import globalquake.core.earthquake.data.Cluster;

import java.util.ArrayList;
import java.util.List;

public class ClusterAnalysisClient extends ClusterAnalysis {

    private final List<Cluster> clusters;

    public ClusterAnalysisClient() {
        clusters = new ArrayList<>();
    }

    @Override
    public List<Cluster> getClusters() {
        return clusters;
    }
}

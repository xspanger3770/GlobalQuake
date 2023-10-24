package globalquake.core.regions;

import globalquake.core.earthquake.ArchivedQuake;
import globalquake.core.earthquake.data.Cluster;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.regions.RegionUpdater;
import globalquake.core.regions.Regional;
import org.junit.Test;

import static org.junit.Assert.*;

public class RegionUpdaterTest {

    @Test
    public void testNonnull(){
        RegionalInst inst = new RegionalInst(50, 17);
        new RegionUpdater(inst);
        assertEquals(RegionUpdater.DEFAULT_REGION, inst.getRegion());
    }


    @Test
    public void testBasicUpdate(){
        RegionalInst inst = new RegionalInst(50, 17);
        RegionUpdater updater = new RegionUpdater(inst);
        updater.updateRegion();
        assertNotEquals(inst.getRegion(), RegionUpdater.DEFAULT_REGION);
    }

    @Test
    public void testEarthquakeRegion(){
        Earthquake earthquake = new Earthquake(new Cluster(0));
        assertNotNull(earthquake.getRegion());
    }

    @Test
    public void testArchivedEarthquakeRegion(){
        ArchivedQuake archivedQuake = new ArchivedQuake(50,17,0,0,0);
        assertNotNull(archivedQuake.getRegion());
    }

    static class RegionalInst implements Regional {

        private final double lat;
        private final double lon;
        private String region;

        public RegionalInst(double lat, double lon){
            this.lat = lat;
            this.lon = lon;
        }

        @Override
        public String getRegion() {
            return region;
        }

        @Override
        public void setRegion(String newRegion) {
            this.region = newRegion;
        }

        @Override
        public double getLat() {
            return lat;
        }

        @Override
        public double getLon() {
            return lon;
        }
    }

}
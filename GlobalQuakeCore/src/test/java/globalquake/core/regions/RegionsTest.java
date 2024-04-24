package globalquake.core.regions;

import globalquake.utils.LookupTableIO;
import org.junit.Test;

import java.io.IOException;
import java.util.HashMap;

import static globalquake.core.regions.Regions.interpolate;
import static org.junit.Assert.*;

public class RegionsTest {

    @Test
    public void basicRegionsTest() throws IOException {
        Regions.init();

        assertEquals("Poland-Czech Republic border region", Regions.getRegion(50, 17));
        assertEquals("Poland", Regions.getRegion(51.8, 18.3));
    }

    @SuppressWarnings("unused")
    public void getShorelineDistanceTest() throws IOException {
        Regions.init();

        double lat = 62.659630,
                lon = -42.440372;

        assertEquals(19.53736973813437, Regions.getShorelineDistance(lat, lon), 0.1);

        lat = 63.185109;
        lon = -44.750247;

        assertEquals(0, Regions.getShorelineDistance(lat, lon), 0);

        lat = 58.664108;
        lon = -67.243589;

        assertEquals(55, Regions.getShorelineDistance(lat, lon), 1);
    }

    @SuppressWarnings("unused")
    public void bilinearInterpolationTest() throws IOException {
        HashMap<String, Double> lookupTable = LookupTableIO.importLookupTableFromFile();
        assertNotNull(lookupTable);

        double interpolation = interpolate(21.673478, -19.158873, lookupTable);
        assertEquals(220, interpolation, 5);

        interpolation = interpolate(-2.376240, -38.963751, lookupTable);
        assertEquals(125, interpolation, 5);
    }

    @SuppressWarnings("unused")
    public void lookupTableEffectivityTest() throws IOException {
        Regions.init();

        HashMap<String, Double> lookupTable = LookupTableIO.importLookupTableFromFile();
        assertNotNull(lookupTable);

        double lat = 62.659630,
                lon = -42.440372;

        double legacyStartTime = System.currentTimeMillis();
        double ignored = Regions.getShorelineDistance(lat, lon);
        double legacyEndTime = System.currentTimeMillis();

        double lookupStartTime = System.currentTimeMillis();
        Regions.interpolate(lat, lon, lookupTable);
        double lookupEndTime = System.currentTimeMillis();

        double legacy = legacyEndTime - legacyStartTime;
        double lookup = lookupEndTime - lookupStartTime;

        assert (lookup < legacy);
    }

    @SuppressWarnings("unused")
    public void isValidPointTest() {
        assertTrue(Regions.isValidPoint(0, 0));
        assertFalse(Regions.isValidPoint(0, -270));
        assertFalse(Regions.isValidPoint(270, 0));
    }

    @SuppressWarnings("unused")
    public void lookupTableGenerationTest() {
        HashMap<String, Double> testLookupTable = Regions.generateLookupTable(0, 1, 0, 1);

        assertEquals(4, testLookupTable.size());

        double lat = 0, lon;
        for (int i = 0; i < 2; i++) {
            lon = 0;

            for (int j = 0; j < 2; j++) {
                String expectedKey = String.format("%.6f,%.6f", lat, lon);
                assertTrue(testLookupTable.containsKey(expectedKey));

                lon += 0.5;
            }
            lat += 0.5;
        }
    }
}
package globalquake.regions;

import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.*;

public class RegionsTest {

    @Test
    public void basicRegionsTest() throws IOException {
        Regions.init();

        assertEquals("Czech Republic", Regions.getRegion(50, 17));
        assertEquals("Poland", Regions.getRegion(51.8, 18.3));
    }

    @Test
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
}
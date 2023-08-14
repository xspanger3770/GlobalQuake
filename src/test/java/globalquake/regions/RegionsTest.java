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

}
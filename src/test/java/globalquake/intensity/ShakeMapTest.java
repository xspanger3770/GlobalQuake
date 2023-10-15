package globalquake.intensity;

import globalquake.core.earthquake.data.Hypocenter;
import globalquake.regions.Regions;
import globalquake.ui.settings.Settings;
import org.junit.Test;

import java.io.IOException;

public class ShakeMapTest {

    @Test
    public void basicTest() throws IOException {
        ShakeMap.init();
        Settings.getSelectedDistanceUnit();
        IntensityTable.init();
        Regions.init();

        Hypocenter hypocenter = new Hypocenter(0,0,0,0,0,0,null,null);
        hypocenter.magnitude = 9.0;

        long a = System.currentTimeMillis();
        ShakeMap shakeMap = new ShakeMap(hypocenter, 5);
        System.err.printf("Shake map generated in %d ms%n", (System.currentTimeMillis() - a));
        System.out.println(shakeMap.getHexList().size());
    }

}
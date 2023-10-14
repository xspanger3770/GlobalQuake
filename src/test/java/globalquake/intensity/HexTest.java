package globalquake.intensity;

import com.uber.h3core.H3Core;

import java.io.IOException;

public class HexTest {

    public static void main(String[] args) throws IOException {
        H3Core h3 = H3Core.newInstance();

        double lat = 50.262;
        double lon = 17.262;
        int res = 4;

        System.out.println(h3.latLngToCellAddress(lat, lon, res));
        System.out.println(h3.gridDisk(h3.latLngToCellAddress(lat, lon, res), 2).size());
    }

}

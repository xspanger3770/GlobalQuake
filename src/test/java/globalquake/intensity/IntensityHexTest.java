package globalquake.intensity;

import org.junit.Test;

import java.util.HashSet;

import static org.junit.Assert.*;

public class IntensityHexTest {

    @Test
    public void hashTest(){
        IntensityHex h1 = new IntensityHex(0, 0, null);
        IntensityHex h2 = new IntensityHex(0, 50, null);

        HashSet<IntensityHex> set = new HashSet<>();
        set.add(h1);
        set.add(h2);

        assertEquals(2, set.size());
    }

}
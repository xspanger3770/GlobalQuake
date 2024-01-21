package globalquake.core.earthquake;

import globalquake.core.earthquake.data.MagnitudeReading;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.*;

public class EarthquakeAnalysisTest {

    @Test
    public void testMagnitudeSelectionEmptyList(){
        List<MagnitudeReading> mags = new ArrayList<>();

        assertEquals(EarthquakeAnalysis.selectMagnitude(mags), EarthquakeAnalysis.NO_MAGNITUDE, 0.1);
    }

    @Test
    public void testMagnitudeSelectionSingleVal(){
        List<MagnitudeReading> mags = new ArrayList<>();

        mags.add(new MagnitudeReading(4,1));

        assertEquals(EarthquakeAnalysis.selectMagnitude(mags), 4.0, 0.1);
    }

    @Test
    public void testMagnitudeSelectionSimple(){
        List<MagnitudeReading> mags = new ArrayList<>();

        mags.add(new MagnitudeReading(4,1));
        mags.add(new MagnitudeReading(4,10));
        mags.add(new MagnitudeReading(4,100));

        assertEquals(EarthquakeAnalysis.selectMagnitude(mags), 4.0, 0.1);
    }

    @Test
    public void testMagnitudeSelectionSort(){
        List<MagnitudeReading> mags = new ArrayList<>();

        mags.add(new MagnitudeReading(1,1));
        mags.add(new MagnitudeReading(2,10));
        mags.add(new MagnitudeReading(3,100));
        mags.add(new MagnitudeReading(4,100));
        mags.add(new MagnitudeReading(5,100));

        Collections.shuffle(mags);

        assertEquals(EarthquakeAnalysis.selectMagnitude(mags), 3.0, 0.1);
    }

    @Test
    public void testMagnitudeSelectionDistant(){
        List<MagnitudeReading> mags = new ArrayList<>();

        mags.add(new MagnitudeReading(4,1));
        mags.add(new MagnitudeReading(4,10));
        mags.add(new MagnitudeReading(4,100));
        mags.add(new MagnitudeReading(9,10_000));
        mags.add(new MagnitudeReading(9,10_000));
        mags.add(new MagnitudeReading(9,10_000));
        mags.add(new MagnitudeReading(9,10_000));
        mags.add(new MagnitudeReading(9,10_000));
        mags.add(new MagnitudeReading(9,10_000));

        Collections.shuffle(mags);

        assertEquals(EarthquakeAnalysis.selectMagnitude(mags), 4.0, 0.1);
    }

    @Test
    public void testMagnitudeSelectionMany(){
        List<MagnitudeReading> mags = new ArrayList<>();

        for(int i = 0; i < 100; i++) {
            mags.add(new MagnitudeReading(4, 1));
        }

        for(int i = 0; i < 100; i++) {
            mags.add(new MagnitudeReading(6, 3000));
        }

        for(int i = 0; i < 100; i++) {
            mags.add(new MagnitudeReading(8, 10000));
        }


        Collections.shuffle(mags);

        assertEquals(EarthquakeAnalysis.selectMagnitude(mags), 4.0, 0.1);
    }

}
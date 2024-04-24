package globalquake.core.geo.taup;

import globalquake.core.exception.FatalApplicationException;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class TauPTravelTimeCalculatorTest {

    @Test
    public void testTravelTimeZeroDistance() throws FatalApplicationException {
        TauPTravelTimeCalculator.init();

        assertEquals(0, Math.abs(TauPTravelTimeCalculator.getPWaveTravelTime(0, 0)), 1e-6);
        assertEquals(0, Math.abs(TauPTravelTimeCalculator.getSWaveTravelTime(0, 0)), 1e-6);
    }

    @Test
    public void testPKPTravelTimeZeroDistance() throws FatalApplicationException {
        TauPTravelTimeCalculator.init();

        assertEquals(TauPTravelTimeCalculator.NO_ARRIVAL, TauPTravelTimeCalculator.getPKPWaveTravelTime(0, 0), 1e-6);
    }

    @Test
    public void sanityTestTravelTime0KM() throws FatalApplicationException {
        TauPTravelTimeCalculator.init();

        double depth = 0;
        for (double ang = 0; ang <= 100; ang += 1) {
            assertEquals("%s˚ %skm".formatted(ang, depth), TravelTimeTableOld.getPWaveTravelTime(depth, ang), TauPTravelTimeCalculator.getPWaveTravelTime(depth, ang), 3);
            assertEquals("%s˚ %skm".formatted(ang, depth), TravelTimeTableOld.getSWaveTravelTime(depth, ang), TauPTravelTimeCalculator.getSWaveTravelTime(depth, ang), 3);
        }
    }

    @Test
    public void sanityTestTravelTimeVarious() throws FatalApplicationException {
        TauPTravelTimeCalculator.init();

        for (double depth = 0; depth < 600; depth += 1) {
            for (double ang = 10; ang <= 100; ang += 1) {
                assertEquals("%s˚ %skm".formatted(ang, depth), TravelTimeTableOld.getPWaveTravelTime(depth, ang), TauPTravelTimeCalculator.getPWaveTravelTime(depth, ang), 2.0);
                assertEquals("%s˚ %skm".formatted(ang, depth), TravelTimeTableOld.getSWaveTravelTime(depth, ang), TauPTravelTimeCalculator.getSWaveTravelTime(depth, ang), 2.0);
            }
        }
    }

    @Test
    public void sanityTestTravelAngle() throws FatalApplicationException {
        TauPTravelTimeCalculator.init();

        for (double depth = 0; depth < 600; depth += 1) {
            for (double ang = 0; ang < 150; ang += 1) {
                assertEquals("%s˚ %skm".formatted(ang, depth), ang, TauPTravelTimeCalculator.getPWaveTravelAngle(depth, TauPTravelTimeCalculator.getPWaveTravelTime(depth, ang)),
                        1e-4);
            }
        }
    }

    @Test
    public void testNoCrash() throws Exception {
        TauPTravelTimeCalculator.init();

        for (double depth = -20; depth <= 1000; depth += 0.04) {
            for (double ang = -20; ang <= 200; ang += 0.04) {
                TauPTravelTimeCalculator.getPWaveTravelTimeFast(depth, ang);
                TauPTravelTimeCalculator.getPWaveTravelTime(depth, ang);
            }
        }
    }

    @Test
    public void testBigAngle() throws Exception {
        TauPTravelTimeCalculator.init();

        assertEquals(TauPTravelTimeCalculator.NO_ARRIVAL, TauPTravelTimeCalculator.getPWaveTravelAngle(0, 40 * 60), 1e-6);
        assertEquals(TauPTravelTimeCalculator.NO_ARRIVAL, TauPTravelTimeCalculator.getPWaveTravelAngle(0, -40 * 60), 1e-6);
        assertEquals(TauPTravelTimeCalculator.NO_ARRIVAL, TauPTravelTimeCalculator.getPKIKPWaveTravelAngle(0, 2 * 60), 1e-6);
        assertEquals(TauPTravelTimeCalculator.NO_ARRIVAL, TauPTravelTimeCalculator.getPKPWaveTravelAngle(0, 2 * 60), 1e-6);
    }

    @Test
    public void testPKP() throws Exception {
        TauPTravelTimeCalculator.init();

        assertEquals(153.35, TauPTravelTimeCalculator.getPKPWaveTravelAngle(0, 1200.0), 0.5);
        assertEquals(1194.46, TauPTravelTimeCalculator.getPKPWaveTravelTime(0, 151.0), 0.5);
        assertEquals(1196.88, TauPTravelTimeCalculator.getPKPWaveTravelTime(0, 152.0), 0.5);
        assertEquals(1199.21, TauPTravelTimeCalculator.getPKPWaveTravelTime(0, 153.0), 0.5);
        assertEquals(1201.45, TauPTravelTimeCalculator.getPKPWaveTravelTime(0, 154.0), 0.5);
        assertEquals(1203.61, TauPTravelTimeCalculator.getPKPWaveTravelTime(0, 155.0), 0.5);
        assertEquals(1222.92, TauPTravelTimeCalculator.getPKPWaveTravelTime(0, 156.0), 0.5);

        assertEquals(151, TauPTravelTimeCalculator.getPKPWaveTravelAngle(0, 1194.46), 0.5);
        assertEquals(152, TauPTravelTimeCalculator.getPKPWaveTravelAngle(0, 1196.88), 0.5);
        assertEquals(153, TauPTravelTimeCalculator.getPKPWaveTravelAngle(0, 1199.21), 0.5);
        assertEquals(154, TauPTravelTimeCalculator.getPKPWaveTravelAngle(0, 1201.45), 0.5);
        assertEquals(155, TauPTravelTimeCalculator.getPKPWaveTravelAngle(0, 1203.61), 0.5);
        assertEquals(156, TauPTravelTimeCalculator.getPKPWaveTravelAngle(0, 1222.92), 0.5);
    }

    @Test
    public void testPKIKP() throws Exception {
        TauPTravelTimeCalculator.init();

        assertEquals(1212.09, TauPTravelTimeCalculator.getPKIKPWaveTravelTime(0, 180.0), 0.5);
        assertEquals(994.57, TauPTravelTimeCalculator.getPKIKPWaveTravelTime(0, 0.0), 0.5);
        assertEquals(999.02, TauPTravelTimeCalculator.getPKIKPWaveTravelTime(0, 20), 0.5);
        assertEquals(1012.15, TauPTravelTimeCalculator.getPKIKPWaveTravelTime(0, 40), 0.5);
        assertEquals(1061.23, TauPTravelTimeCalculator.getPKIKPWaveTravelTime(0, 80), 0.5);
        assertEquals(1186.73, TauPTravelTimeCalculator.getPKIKPWaveTravelTime(0, 150), 0.5);

        assertEquals(20, TauPTravelTimeCalculator.getPKIKPWaveTravelAngle(0, 999.02), 0.5);
        assertEquals(0, TauPTravelTimeCalculator.getPKIKPWaveTravelAngle(0, 994.47), 0.5);
        assertEquals(40, TauPTravelTimeCalculator.getPKIKPWaveTravelAngle(0, 1012.15), 0.5);
        assertEquals(80, TauPTravelTimeCalculator.getPKIKPWaveTravelAngle(0, 1061.23), 0.5);
        assertEquals(150, TauPTravelTimeCalculator.getPKIKPWaveTravelAngle(0, 1186.73), 0.5);
        assertEquals(179.9, TauPTravelTimeCalculator.getPKIKPWaveTravelAngle(0, 1212.09), 0.5);
    }

}
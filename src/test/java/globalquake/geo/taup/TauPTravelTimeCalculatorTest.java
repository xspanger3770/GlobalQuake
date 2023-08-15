package globalquake.geo.taup;

import globalquake.exception.FatalApplicationException;
import org.junit.Test;

import static org.junit.Assert.*;

public class TauPTravelTimeCalculatorTest {

    @Test
    public void testTravelTimeZeroDistance() throws FatalApplicationException {
        TauPTravelTimeCalculator.init();

        assertEquals(0, Math.abs(TauPTravelTimeCalculator.getPWaveTravelTime(0, 0)) ,1e-6);
        assertEquals(0, Math.abs(TauPTravelTimeCalculator.getSWaveTravelTime(0, 0)) , 1e-6);
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

        for(double depth = 0; depth < 600; depth += 1) {
            for (double ang = 10; ang <= 100; ang += 1) {
                assertEquals("%s˚ %skm".formatted(ang, depth), TravelTimeTableOld.getPWaveTravelTime(depth, ang), TauPTravelTimeCalculator.getPWaveTravelTime(depth, ang), 2.0);
                assertEquals("%s˚ %skm".formatted(ang, depth), TravelTimeTableOld.getSWaveTravelTime(depth, ang), TauPTravelTimeCalculator.getSWaveTravelTime(depth, ang), 2.0);
            }
        }
    }

    @Test
    public void sanityTestTravelAngle() throws FatalApplicationException {
        TauPTravelTimeCalculator.init();

        for(double depth = 0; depth < 600; depth += 1) {
            for (double ang = 0; ang < 150; ang += 1) {
                assertEquals("%s˚ %skm".formatted(ang, depth), ang, TauPTravelTimeCalculator.getPWaveTravelAngle(depth, TauPTravelTimeCalculator.getPWaveTravelTime(depth, ang)),
                         0.01);
            }
        }
    }

}
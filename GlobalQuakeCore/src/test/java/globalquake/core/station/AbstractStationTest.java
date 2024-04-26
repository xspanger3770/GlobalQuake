package globalquake.core.station;

import gqserver.api.packets.station.InputType;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class AbstractStationTest {

    @Test
    public void testIntervals() {
        AbstractStation abstractStation = new GlobalStation("", "", "", "", 5, 5, 5, 5, null, -1, InputType.UNKNOWN);
        StationInterval int1 = new StationInterval(10, 50, StationState.INACTIVE);
        StationInterval int2 = new StationInterval(50, 70, StationState.ACTIVE);
        StationInterval int3 = new StationInterval(100, 150, StationState.ACTIVE);
        abstractStation.getIntervals().add(int1);
        abstractStation.getIntervals().add(int2);
        abstractStation.getIntervals().add(int3);
        assertEquals(StationState.UNKNOWN, abstractStation.getStateAt(0));
        assertEquals(StationState.INACTIVE, abstractStation.getStateAt(10));
        assertEquals(StationState.ACTIVE, abstractStation.getStateAt(50));
        assertEquals(StationState.UNKNOWN, abstractStation.getStateAt(80));
        assertEquals(StationState.ACTIVE, abstractStation.getStateAt(120));
    }

    @Test
    public void testReport() {
        AbstractStation abstractStation = new GlobalStation("", "", "", "", 5, 5, 5, 5, null, -1, InputType.UNKNOWN);
        abstractStation.reportState(StationState.ACTIVE, 0);
        abstractStation.reportState(StationState.ACTIVE, 10);

        assertEquals(StationState.UNKNOWN, abstractStation.getStateAt(-10));
        assertEquals(StationState.ACTIVE, abstractStation.getStateAt(0));
        assertEquals(StationState.ACTIVE, abstractStation.getStateAt(5));
        assertEquals(StationState.UNKNOWN, abstractStation.getStateAt(10));
        assertEquals(StationState.UNKNOWN, abstractStation.getStateAt(20));

        abstractStation.reportState(StationState.ACTIVE, 50);
        assertEquals(StationState.ACTIVE, abstractStation.getStateAt(5));
        assertEquals(StationState.ACTIVE, abstractStation.getStateAt(10));
        assertEquals(StationState.ACTIVE, abstractStation.getStateAt(20));
        assertEquals(StationState.ACTIVE, abstractStation.getStateAt(49));


        abstractStation.reportState(StationState.ACTIVE, 60 + AbstractStation.INTERVAL_MAX_GAP);
        abstractStation.reportState(StationState.ACTIVE, 70 + AbstractStation.INTERVAL_MAX_GAP);
        assertEquals(StationState.UNKNOWN, abstractStation.getStateAt(50));
        assertEquals(StationState.UNKNOWN, abstractStation.getStateAt(60));
        assertEquals(StationState.ACTIVE, abstractStation.getStateAt(60 + AbstractStation.INTERVAL_MAX_GAP));
        assertEquals(StationState.ACTIVE, abstractStation.getStateAt(65 + AbstractStation.INTERVAL_MAX_GAP));

        abstractStation.reportState(StationState.ACTIVE, 51 + AbstractStation.INTERVAL_STORAGE_TIME);
        assertEquals(StationState.UNKNOWN, abstractStation.getStateAt(-10));
        assertEquals(StationState.UNKNOWN, abstractStation.getStateAt(0));
        assertEquals(StationState.UNKNOWN, abstractStation.getStateAt(5));
        assertEquals(StationState.UNKNOWN, abstractStation.getStateAt(10));
        assertEquals(StationState.UNKNOWN, abstractStation.getStateAt(20));
        assertEquals(StationState.ACTIVE, abstractStation.getStateAt(60 + AbstractStation.INTERVAL_MAX_GAP));
        assertEquals(StationState.ACTIVE, abstractStation.getStateAt(65 + AbstractStation.INTERVAL_MAX_GAP));
    }

}
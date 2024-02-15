package globalquake.core.analysis;

import org.junit.Test;

import static org.junit.Assert.*;

public class WaveformBufferTest {

    @Test
    public void testSize() {
        double sps = 30.0;
        int seconds = 10;
        WaveformBuffer waveformBuffer = new WaveformBuffer(sps, seconds, false);
        assertEquals(300, waveformBuffer.getSize());
    }

    @Test
    public void testRing() {
        double sps = 1.0;
        int seconds = 10;
        WaveformBuffer waveformBuffer = new WaveformBuffer(sps, seconds, false);
        assertEquals(10, waveformBuffer.getSize());
        assertEquals(0, waveformBuffer.getNextSlot());
        assertEquals(0, waveformBuffer.getOldestDataSlot());
        assertTrue(waveformBuffer.isEmpty());

        waveformBuffer.log(0, 0, 0, 0, 0, 1, 0, false);
        assertEquals(1, waveformBuffer.getNextSlot());
        assertEquals(0, waveformBuffer.getOldestDataSlot());

        for (int i = 0; i < 9; i++) {
            waveformBuffer.log(i + 1, 0, 0, 0, 0, 1, 0, false);
        }
        assertEquals(waveformBuffer.getNextSlot(), 0);
        assertEquals(waveformBuffer.getOldestDataSlot(), 0);
        waveformBuffer.log(10, 0, 0, 0, 0, 1, 0, false);
        assertEquals(waveformBuffer.getNextSlot(), 1);
        assertEquals(waveformBuffer.getOldestDataSlot(), 1);
    }

    @Test
    public void testStorage() {
        double sps = 1.0;
        int seconds = 10;
        WaveformBuffer waveformBuffer = new WaveformBuffer(sps, seconds, false);
        waveformBuffer.log(0, 10, 10, 10, 10, 1, 10, false);

        Log log0 = waveformBuffer.toLog(0);

        assertEquals(log0.time(), 0);
        assertEquals(log0.ratio(), 10, 1e-6);
    }

    @Test
    public void testResize() {
        double sps = 1.0;
        int seconds = 10;
        WaveformBuffer waveformBuffer = new WaveformBuffer(sps, seconds, false);

        for (int i = 0; i < waveformBuffer.getSize(); i++) {
            waveformBuffer.log(i * 1000L, i * 10, i * 20, i * 30, i * 40, 1, i * 60, false);
        }

        waveformBuffer.resize(3);

        assertEquals(3, waveformBuffer.getSize());

        Log log0 = waveformBuffer.toLog(waveformBuffer.getOldestDataSlot());

        assertEquals(7000, log0.time());
        assertEquals(30 * 7, log0.ratio(), 1e-6);

        waveformBuffer.log(10000, 100, 100, 100, 100, 1, 100, false);

        log0 = waveformBuffer.toLog(waveformBuffer.getOldestDataSlot());

        assertEquals(8000, log0.time());
        assertEquals(30 * 8, log0.ratio(), 1e-6);

        log0 = waveformBuffer.toLog(waveformBuffer.getNewestDataSlot());

        assertEquals(10000, log0.time());
        assertEquals(100, log0.ratio(), 1e-6);
    }

    @Test
    public void testExpand() {
        WaveformBuffer waveformBuffer = new WaveformBuffer(1, 3, false);
        assertEquals(3, waveformBuffer.getSize());
        for (int i = 0; i < waveformBuffer.getSize(); i++) {
            waveformBuffer.log(i * 1000L, i, i, i, i, i, i, false);
        }

        assertEquals(0, waveformBuffer.getTime(waveformBuffer.getOldestDataSlot()));
        waveformBuffer.log(3000, 3, 3, 3, 3, 3, 3, true);
        assertEquals(6, waveformBuffer.getSize());
        assertEquals(0, waveformBuffer.getTime(waveformBuffer.getOldestDataSlot()));
        assertEquals(3000, waveformBuffer.getTime(waveformBuffer.getNewestDataSlot()));

    }

}
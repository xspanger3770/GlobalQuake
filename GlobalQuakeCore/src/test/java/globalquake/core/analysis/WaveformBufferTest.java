package globalquake.core.analysis;

import org.junit.Test;

import static org.junit.Assert.*;

public class WaveformBufferTest {

    @Test
    public void testSize(){
        double sps = 30.0;
        int seconds = 10;
        WaveformBuffer waveformBuffer = new WaveformBuffer(sps, seconds);
        assertEquals(300, waveformBuffer.getSize());
    }

    @Test
    public void testRing(){
        double sps = 1.0;
        int seconds = 10;
        WaveformBuffer waveformBuffer = new WaveformBuffer(sps, seconds);
        assertEquals(10, waveformBuffer.getSize());
        assertEquals(0, waveformBuffer.getNextSlot());
        assertEquals(0, waveformBuffer.getOldestDataSlot());
        assertTrue(waveformBuffer.isEmpty());

        waveformBuffer.log(0,0,0,0,0,0,0);
        assertEquals(1, waveformBuffer.getNextSlot());
        assertEquals(0, waveformBuffer.getOldestDataSlot());

        for(int i = 0; i < 9; i++){
            waveformBuffer.log(i + 1,0,0,0,0,0,0);
        }
        assertEquals(waveformBuffer.getNextSlot(), 0);
        assertEquals(waveformBuffer.getOldestDataSlot(), 0);
        waveformBuffer.log(10,0,0,0,0,0,0);
        assertEquals(waveformBuffer.getNextSlot(), 1);
        assertEquals(waveformBuffer.getOldestDataSlot(), 1);
    }

    @Test
    public void testStorage(){
        double sps = 1.0;
        int seconds = 10;
        WaveformBuffer waveformBuffer = new WaveformBuffer(sps, seconds);
        waveformBuffer.log(0,10,10,10,10,10,10);

        Log log0 = waveformBuffer.toLog(0);

        assertEquals(log0.time(), 0);
        assertEquals(log0.shortAverage(), 10, 1e-6);
    }

    @Test
    public void testResize(){
        double sps = 1.0;
        int seconds = 10;
        WaveformBuffer waveformBuffer = new WaveformBuffer(sps, seconds);

        for(int i = 0; i < waveformBuffer.getSize(); i++) {
            waveformBuffer.log(i, i*10, i*20, i*30, i*40, i*50, i*60);
        }

        waveformBuffer.resize(3);

        assertEquals(3, waveformBuffer.getSize());

        Log log0 = waveformBuffer.toLog(waveformBuffer.getOldestDataSlot());

        assertEquals(7, log0.time());
        assertEquals(30*7, log0.shortAverage(), 1e-6);

        waveformBuffer.log(100,100,100,100,100,100,100);

        log0 = waveformBuffer.toLog(waveformBuffer.getOldestDataSlot());

        assertEquals(8, log0.time());
        assertEquals(30*8, log0.shortAverage(), 1e-6);

        log0 = waveformBuffer.toLog(waveformBuffer.getNewestDataSlot());

        assertEquals(100, log0.time());
        assertEquals(100, log0.shortAverage(), 1e-6);
    }

}
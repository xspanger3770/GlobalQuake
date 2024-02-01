package globalquake.core.analysis;

import org.junit.Test;

import static org.junit.Assert.*;

public class WaveformBufferTest {

    @Test
    public void testSize(){
        double sps = 30.0;
        int seconds = 10;
        WaveformBuffer waveformBuffer = new WaveformBuffer(sps, seconds);
        assertEquals(waveformBuffer.getSize(), 300);
    }

    @Test
    public void testRing(){
        double sps = 1.0;
        int seconds = 10;
        WaveformBuffer waveformBuffer = new WaveformBuffer(sps, seconds);
        assertEquals(waveformBuffer.getSize(), 10);
        assertEquals(waveformBuffer.getNextFreeSlot(), 0);
        assertEquals(waveformBuffer.getOldestDataSlot(), 9);
        assertTrue(waveformBuffer.isEmpty());

        waveformBuffer.log(0,0,0,0,0,0,0);
        assertEquals(waveformBuffer.getNextFreeSlot(), 1);
        assertEquals(waveformBuffer.getOldestDataSlot(), 9);

        for(int i = 0; i < 9; i++){
            waveformBuffer.log(i + 1,0,0,0,0,0,0);
        }
        assertEquals(waveformBuffer.getNextFreeSlot(), 0);
        assertEquals(waveformBuffer.getOldestDataSlot(), 0);
        waveformBuffer.log(10,0,0,0,0,0,0);
        assertEquals(waveformBuffer.getNextFreeSlot(), 1);
        assertEquals(waveformBuffer.getOldestDataSlot(), 1);
    }

}
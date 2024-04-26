package globalquake.sounds;

public class SoundsInfo {

    public int maxLevel = -1;
    public double maxPGA;
    public double maxPGAHome;
    public boolean warningPlayed;
    public int lastCountdown = 999;
    public boolean meets;

    public final long createdAt = System.currentTimeMillis();

}

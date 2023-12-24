package globalquake.sounds;

public class SoundsInfo {

	public boolean firstSound;
	public int maxLevel;
	public double maxPGA;
	public double maxPGAHome;
	public boolean warningPlayed;
	public int lastCountdown = 999;
	public boolean meets;

	public final long createdAt = System.currentTimeMillis();

}

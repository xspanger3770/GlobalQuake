package globalquake.sounds;

public class SoundsInfo {

	public boolean firstSound;
	public int maxLevel;
	public double maxPGA;
	public double maxPGAHome;
	public boolean warningPlayed;
	public int lastCountdown = -1;
	public boolean meets;

	public long createdAt = System.currentTimeMillis();

}

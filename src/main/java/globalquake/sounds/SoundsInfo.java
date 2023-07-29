package globalquake.sounds;

public class SoundsInfo {

	public boolean firstSound;
	public int maxLevel;
	public double maxPGA;
	public double maxPGAHome;
	public boolean warningPlayed;
	public int lastCountdown;
	public boolean meets;

	public SoundsInfo() {
		this.lastCountdown = -1;
	}

}

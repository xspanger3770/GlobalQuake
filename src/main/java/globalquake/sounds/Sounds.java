package globalquake.sounds;

import globalquake.core.AlertManager;
import globalquake.core.earthquake.data.Cluster;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.exception.FatalIOException;
import globalquake.geo.GeoUtils;
import globalquake.intensity.ShindoIntensityScale;
import globalquake.ui.settings.Settings;

import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.Clip;
import java.util.Objects;

public class Sounds {

	public static Clip weak;
	public static Clip shindo1;
	public static Clip shindo5;
	public static Clip warning;
	public static Clip incoming;
	public static Clip eew;
	public static Clip eew_warning;
	public static Clip felt;
	public static Clip dong;

	public static boolean soundsAvailable = true;

	private static Clip loadSound(String res) throws FatalIOException {
		try {
			AudioInputStream audioIn = AudioSystem.getAudioInputStream(Objects.requireNonNull(ClassLoader.getSystemClassLoader().getResource(res)));
			Clip clip = AudioSystem.getClip();
			clip.open(audioIn);
			return clip;
		} catch(Exception e){
			soundsAvailable = false;
			throw new FatalIOException("Failed to load sound: "+res, e);
		}
	}

	public static void load() throws Exception {
		weak = loadSound("sounds/weak.wav");
		shindo1 = loadSound("sounds/shindo1.wav");
		shindo5 = loadSound("sounds/shindo5.wav");
		incoming = loadSound("sounds/incoming.wav");
		eew = loadSound("sounds/eew.wav");
		warning = loadSound("sounds/warning.wav");
		eew_warning = loadSound("sounds/eew_warning.wav");
		felt = loadSound("sounds/felt.wav");
		dong = loadSound("sounds/dong.wav");

	}

	public static void playSound(Clip clip) {
		if(!Settings.enableSound || !soundsAvailable || clip == null) {
			return;
		}

		clip.setFramePosition(0);
		clip.start();
	}

	public static void determineSounds(Cluster cluster) {
		SoundsInfo info = cluster.soundsInfo;

		if (!info.firstSound) {
			Sounds.playSound(Sounds.weak);
			info.firstSound = true;
		}

		int level = cluster.getActualLevel();
		if (level > info.maxLevel) {
			if (level >= 1 && info.maxLevel < 1) {
				Sounds.playSound(Sounds.shindo1);
			}
			if (level >= 2 && info.maxLevel < 2) {
				Sounds.playSound(Sounds.shindo5);
			}
			if (level >= 3 && info.maxLevel < 3) {
				Sounds.playSound(Sounds.warning);
			}
			info.maxLevel = level;
		}
		Earthquake quake = cluster.getEarthquake();

		if (quake != null) {
			boolean meets = AlertManager.meetsConditions(quake);
			if (meets && !info.meets) {
				Sounds.playSound(Sounds.eew);
				info.meets = true;
			}
			double pga = GeoUtils.pgaFunctionGen1(cluster.getEarthquake().getMag(), cluster.getEarthquake().getDepth());
			if (info.maxPGA < pga) {

				info.maxPGA = pga;
				if (info.maxPGA >= 100 && !info.warningPlayed && level >= 2) {
					Sounds.playSound(Sounds.eew_warning);
					info.warningPlayed = true;
				}
			}

			double distGEO = GeoUtils.geologicalDistance(quake.getLat(), quake.getLon(), -quake.getDepth(),
					Settings.homeLat, Settings.homeLon, 0.0);
			double pgaHome = GeoUtils.pgaFunctionGen1(quake.getMag(), distGEO);

			if (info.maxPGAHome < pgaHome) {
				if (pgaHome >= ShindoIntensityScale.ICHI.getPga() && info.maxPGAHome < ShindoIntensityScale.ICHI.getPga()) {
					Sounds.playSound(Sounds.felt);
				}
				info.maxPGAHome = pgaHome;
			}

			if (info.maxPGAHome >= ShindoIntensityScale.ICHI.getPga()) {
				if (info.lastCountdown == 0) {
					info.lastCountdown = -999;
					Sounds.playSound(Sounds.dong);
				}
			}
		}
	}
	
}

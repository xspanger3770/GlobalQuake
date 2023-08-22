package globalquake.sounds;

import globalquake.core.AlertManager;
import globalquake.core.earthquake.Cluster;
import globalquake.core.earthquake.Earthquake;
import globalquake.exception.FatalIOException;
import globalquake.geo.GeoUtils;
import globalquake.geo.Level;
import globalquake.geo.Shindo;
import globalquake.geo.taup.TauPTravelTimeCalculator;
import globalquake.ui.settings.Settings;
import org.tinylog.Logger;

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

	public static final int[] countdown_levels = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 120, 180 };

	public static final Clip[] levelsFirst = new Clip[10];
	public static final Clip[] levelsNext = new Clip[9];
	public static final Clip[] countdowns = new Clip[countdown_levels.length];

	public static final boolean soundsEnabled = true;
	public static boolean soundsAvailable = true;

	private static final String[] shindoNames = { "0", "1", "2", "3", "4", "5minus", "5plus", "6minus", "6plus", "7" };

	public static final boolean ENABLE_EXTREME_ALARMS = false;

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

		for (int i = 0; i < shindoNames.length; i++) {
			Clip first = loadSound("sounds/levels/level_" + shindoNames[i] + ".wav");
			levelsFirst[i] = first;

			if (i != shindoNames.length - 1) {
				Clip next = loadSound("sounds/levels/up_to_" + shindoNames[i + 1] + ".wav");
				levelsNext[i] = next;
			}
		}

		for (int i = 0; i < countdown_levels.length; i++) {
			int j = countdown_levels[i];
			String str = j < 10 ? "0" + j : j + "";
			Clip count = loadSound("sounds/countdown/countdown_" + str + ".wav");
			countdowns[i] = count;
		}
	}

	public static Clip nextLevelBeginsWith1(int i) {
		return levelsNext[i];
	}

	public static void playSound(Clip clip) {
		if(!Settings.enableSound || !soundsAvailable) {
			Logger.debug(clip.toString() + " not played. Sound disabled.");
			return;
		}
		if (soundsEnabled && clip != null) {
			clip.setFramePosition(0);
			clip.start();
		}
	}

	public static int getLastCountdown(int secondsS) {
		for (int i = 0; i <= countdown_levels.length - 1; i++) {
			if (countdown_levels[i] >= secondsS) {
				return i;
			}
		}
		return -1;
	}

	public static void determineSounds(Cluster c) {
		SoundsInfo info = c.soundsInfo;

		if (!info.firstSound) {
			Sounds.playSound(Sounds.weak);
			info.firstSound = true;
		}

		int level = c.getActuallLevel();
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
		Earthquake quake = c.getEarthquake();

		if (quake != null) {
			boolean meets = AlertManager.meetsConditions(quake);
			if (meets && !info.meets) {
				Sounds.playSound(Sounds.eew);
				info.meets = true;
			}
			double pga = GeoUtils.pgaFunctionGen1(c.getEarthquake().getMag(), c.getEarthquake().getDepth());
			if (info.maxPGA < pga) {

				info.maxPGA = pga;
				if (info.maxPGA >= 100 && !info.warningPlayed && level >= 2) {
					Sounds.playSound(Sounds.eew_warning);
					info.warningPlayed = true;
				}
			}

			double distGEO = GeoUtils.geologicalDistance(quake.getLat(), quake.getLon(), -quake.getDepth(),
					Settings.homeLat, Settings.homeLon, 0.0);
			double distGC = GeoUtils.greatCircleDistance(quake.getLat(), quake.getLon(), Settings.homeLat,
					Settings.homeLon);
			double pgaHome = GeoUtils.pgaFunctionGen1(quake.getMag(), distGEO);

			if (info.maxPGAHome < pgaHome) {
				Level shindoLast = Shindo.getLevel(info.maxPGAHome);
				Level shindoNow = Shindo.getLevel(pgaHome);
				if (shindoLast != shindoNow && (shindoNow != null ? shindoNow.index() : 0) > 0 && ENABLE_EXTREME_ALARMS) {
					Sounds.playSound(Sounds.nextLevelBeginsWith1(shindoNow.index() - 1));
				}

				if (pgaHome >= Shindo.ZERO.pga() && info.maxPGAHome < Shindo.ZERO.pga()) {
					Sounds.playSound(Sounds.felt);
				}
				info.maxPGAHome = pgaHome;
			}

			if (info.maxPGAHome >= Shindo.ZERO.pga()) {
				double age = (System.currentTimeMillis() - quake.getOrigin()) / 1000.0;

				double sTravel = (long) (TauPTravelTimeCalculator.getSWaveTravelTime(quake.getDepth(),
						TauPTravelTimeCalculator.toAngle(distGC)));
				int secondsS = (int) Math.max(0, Math.ceil(sTravel - age));

				int soundIndex = -1;

				if (info.lastCountdown == -1) {
					soundIndex = Sounds.getLastCountdown(secondsS);
				} else {
					int si = Sounds.getLastCountdown(secondsS);
					if (si < info.lastCountdown) {
						soundIndex = si;
					}
				}

				if (info.lastCountdown == 0) {
					info.lastCountdown = -999;
					Sounds.playSound(Sounds.dong);
				}

				if (soundIndex != -1) {
					if(ENABLE_EXTREME_ALARMS) {
						Sounds.playSound(Sounds.countdowns[soundIndex]);
					}
					info.lastCountdown = soundIndex;
				}
			}
		}
	}
	
}

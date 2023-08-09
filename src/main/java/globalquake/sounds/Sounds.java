package globalquake.sounds;

import globalquake.exception.FatalIOException;
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

	public static final int[] countdown_levels = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 120, 180 };

	public static final Clip[] levelsFirst = new Clip[10];
	public static final Clip[] levelsNext = new Clip[9];
	public static final Clip[] countdowns = new Clip[countdown_levels.length];

	public static final boolean soundsEnabled = true;

	private static final String[] shindoNames = { "0", "1", "2", "3", "4", "5minus", "5plus", "6minus", "6plus", "7" };

	private static Clip loadSound(String res) throws FatalIOException {
		try {
			AudioInputStream audioIn = AudioSystem.getAudioInputStream(Objects.requireNonNull(ClassLoader.getSystemClassLoader().getResource(res)));
			Clip clip = AudioSystem.getClip();
			clip.open(audioIn);
			return clip;
		} catch(Exception e){
			throw new FatalIOException("Cannot load sound: "+res, e);
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
		if(!Settings.enableSound) {
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
	
}

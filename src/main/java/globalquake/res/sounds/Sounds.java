package globalquake.res.sounds;

import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.Clip;

import globalquake.res.sounds.countdown.Countdown;
import globalquake.res.sounds.levels.Levels;
import globalquake.settings.Settings;

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

	public static Clip[] levelsFirst = new Clip[10];
	public static Clip[] levelsNext = new Clip[9];
	public static Clip[] countdowns = new Clip[countdown_levels.length];

	public static boolean soundsEnabled = true;

	private static String[] strs = { "0", "1", "2", "3", "4", "5minus", "5plus", "6minus", "6plus", "7" };
	static {
		try {
			AudioInputStream audioIn = AudioSystem.getAudioInputStream(Sounds.class.getResource("weak.wav"));
			weak = AudioSystem.getClip();
			weak.open(audioIn);
			audioIn = AudioSystem.getAudioInputStream(Sounds.class.getResource("shindo1.wav"));
			shindo1 = AudioSystem.getClip();
			shindo1.open(audioIn);
			audioIn = AudioSystem.getAudioInputStream(Sounds.class.getResource("shindo5.wav"));
			shindo5 = AudioSystem.getClip();
			shindo5.open(audioIn);
			audioIn = AudioSystem.getAudioInputStream(Sounds.class.getResource("incoming.wav"));
			incoming = AudioSystem.getClip();
			incoming.open(audioIn);
			audioIn = AudioSystem.getAudioInputStream(Sounds.class.getResource("eew.wav"));
			eew = AudioSystem.getClip();
			eew.open(audioIn);
			audioIn = AudioSystem.getAudioInputStream(Sounds.class.getResource("warning.wav"));
			warning = AudioSystem.getClip();
			warning.open(audioIn);
			audioIn = AudioSystem.getAudioInputStream(Sounds.class.getResource("eew_warning.wav"));
			eew_warning = AudioSystem.getClip();
			eew_warning.open(audioIn);
			audioIn = AudioSystem.getAudioInputStream(Sounds.class.getResource("felt.wav"));
			felt = AudioSystem.getClip();
			felt.open(audioIn);
			audioIn = AudioSystem.getAudioInputStream(Sounds.class.getResource("dong.wav"));
			dong = AudioSystem.getClip();
			dong.open(audioIn);

			for (int i = 0; i < strs.length; i++) {
				try {
					audioIn = AudioSystem.getAudioInputStream(Levels.class.getResource("level_" + strs[i] + ".wav"));
					Clip first = AudioSystem.getClip();
					first.open(audioIn);

					levelsFirst[i] = first;

					if (i != strs.length - 1) {
						audioIn = AudioSystem
								.getAudioInputStream(Levels.class.getResource("up_to_" + strs[i + 1] + ".wav"));
						Clip next = AudioSystem.getClip();
						next.open(audioIn);
						levelsNext[i] = next;
					}
				} catch (Exception e) {
					e.printStackTrace();
				}
			}

			for (int i = 0; i < countdown_levels.length; i++) {
				int j = countdown_levels[i];
				String str = j < 10 ? "0" + j : j + "";
				System.out.println("countdown_" + str + ".wav");
				audioIn = AudioSystem.getAudioInputStream(Countdown.class.getResource("countdown_" + str + ".wav"));
				Clip count = AudioSystem.getClip();
				count.open(audioIn);
				countdowns[i] = count;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static Clip firstLevelBeginsWith0(int i) {
		return levelsFirst[i];
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

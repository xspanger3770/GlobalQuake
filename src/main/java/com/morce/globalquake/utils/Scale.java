package com.morce.globalquake.utils;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.jtransforms.fft.DoubleFFT_1D;

import com.morce.globalquake.res.Res;

public class Scale {

	private static BufferedImage pgaScale;
	static {
		try {
			pgaScale = ImageIO.read(Res.class.getResource("pgaScale3.png"));
		} catch (IOException e) {

			e.printStackTrace();
		}
	}

	public static Color getColorPGA(double pga) {
		int i = (int) (Math.log10(pga * 100.0) * 20.0);
		return new Color(pgaScale.getRGB(0, Math.max(0, Math.min(pgaScale.getHeight() - 1, i))));
	}

	public static Color getColorRatio(double ratio) {
		int i = (int) (Math.log10(ratio) * 20.0);
		return new Color(pgaScale.getRGB(0, Math.max(0, Math.min(pgaScale.getHeight() - 1, i))));
	}

	public static Color getColorEasily(double ratio) {
		int i = (int) ((pgaScale.getHeight() - 1) * ratio);
		return new Color(pgaScale.getRGB(0, Math.max(0, Math.min(pgaScale.getHeight() - 1, i))));
	}

	public static void main(String[] args) {
		int sps = 50;
		int window = sps * 10; // 10 second window
		double[] data = new double[window];

		double freq = 3;
		for (int i = 0; i < data.length; i++) {
			double v = Math.cos((i / (double) sps) * freq * Math.PI * 2);
			data[i] = v;
		}

		double[] magnitude = new double[window / 2];
		double[] fft = new double[window * 2];
		for (int i = 0; i <= window - 1; i++) {
			fft[2 * i] = data[i];
			fft[2 * i + 1] = 0;
		}

		DoubleFFT_1D ifft = new DoubleFFT_1D(window);
		ifft.complexForward(fft);

		double maxMag = 0;

		for (int i = 0; i < window / 2 - 1; i++) {
			double re = fft[2 * i];
			double im = fft[2 * i + 1];
			double mag = Math.sqrt(re * re + im * im);
			magnitude[i] = mag;
			if (mag > maxMag) {
				maxMag = mag;
			}
		}

		for (int i = 0; i < magnitude.length; i++) {
			double d = magnitude[i];
			// freq = sps * (i / window)
			double freqq = sps * (i / (double) window);
			System.out.println(freqq + "Hz, " + d);
		}
		System.out.println("maxMag = " + maxMag);
	}

	public static Color getColorLevel(int level) {
		if(level==4) {
			return Color.magenta;
		}
		if(level==3) {
			return Color.red;
		}
		if(level==2) {
			return Color.yellow;
		}
		if(level==1) {
			return Color.green;
		}
		return Color.white;
	}

	

}

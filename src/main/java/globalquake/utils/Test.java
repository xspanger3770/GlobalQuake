package globalquake.utils;

import globalquake.geo.TravelTimeTable;

import java.util.Random;

public class Test {

	static Random r = new Random();

	public static void main(String[] args) throws Exception {
		System.out.println(TravelTimeTable.getEpicenterDistance(0, 2));
	}

}

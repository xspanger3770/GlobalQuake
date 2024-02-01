package globalquake.core.analysis;

public record Log(long time, int rawValue, float filteredV, float shortAverage, float mediumAverage, float longAverage,
				  float specialAverage) {

	public double getRatio() {
		return shortAverage() / longAverage();
	}

	public double getMediumRatio() {
		return mediumAverage() / longAverage();
	}

	public double getSpecialRatio() {
		return specialAverage() / longAverage();
	}

}

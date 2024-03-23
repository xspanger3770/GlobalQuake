package globalquake.core.database;

public class SensitivityCorrection {

    private String networkCode;
    private String stationCode;
    private double multiplier;

    public SensitivityCorrection(String networkCode, String stationCode, double multiplier) {
        this.networkCode = networkCode;
        this.stationCode = stationCode;
        this.multiplier = multiplier;
    }

    public double getMultiplier() {
        return multiplier;
    }

    public boolean match(String networkCode, String stationCode){
        if(networkCode.toUpperCase().startsWith(this.networkCode.toUpperCase()) && stationCode.toUpperCase().startsWith(this.stationCode.toUpperCase())){
            return true;
        }

        return false;
    }
}

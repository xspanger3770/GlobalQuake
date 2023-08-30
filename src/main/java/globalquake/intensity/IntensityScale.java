package globalquake.intensity;

import java.util.List;

public interface IntensityScale {

    /**
     * @return List of levels sorted from biggest to smallest
     */
    List<Level> getLevels();

    double getDarkeningFactor();

    default Level getLevel(double pga){
        for(Level level : getLevels()){
            if(pga > level.getPga()){
                return level;
            }
        }

        return null;
    }
}

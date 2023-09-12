package globalquake.ui.archived;

import globalquake.core.earthquake.ArchivedQuake;
import globalquake.ui.globe.GlobePanel;

import javax.swing.*;
import java.awt.*;

public class ArchivedQuakePanel extends GlobePanel {
    public ArchivedQuakePanel(Frame parent, ArchivedQuake quake) {
        super(quake.getLat(), quake.getLon());
        setPreferredSize(new Dimension(600,480));
        setCinemaMode(true);
    }
}

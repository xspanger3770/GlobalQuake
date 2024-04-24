package gqserver.ui.server.tabs;

import globalquake.core.events.GlobalQuakeEventListener;
import globalquake.core.events.specific.QuakeArchiveEvent;
import globalquake.core.events.specific.QuakeCreateEvent;
import globalquake.core.events.specific.QuakeRemoveEvent;
import globalquake.core.events.specific.QuakeUpdateEvent;
import gqserver.server.GlobalQuakeServer;
import gqserver.ui.server.table.GQTable;
import gqserver.ui.server.table.model.EarthquakeTableModel;

import javax.swing.*;
import java.awt.*;

public class EarthquakesTab extends JPanel {

    public EarthquakesTab() {
        setLayout(new BorderLayout());

        EarthquakeTableModel model;
        add(new JScrollPane(new GQTable<>(
                model = new EarthquakeTableModel(GlobalQuakeServer.instance.getEarthquakeAnalysis().getEarthquakes()))));

        GlobalQuakeServer.instance.getEventHandler().registerEventListener(new GlobalQuakeEventListener() {
            @Override
            public void onQuakeUpdate(QuakeUpdateEvent event) {
                model.applyFilter();
            }

            @Override
            public void onQuakeRemove(QuakeRemoveEvent quakeRemoveEvent) {
                model.applyFilter();
            }

            @Override
            public void onQuakeArchive(QuakeArchiveEvent quakeArchiveEvent) {
                model.applyFilter();
            }

            @Override
            public void onQuakeCreate(QuakeCreateEvent event) {
                model.applyFilter();
            }
        });
    }

}

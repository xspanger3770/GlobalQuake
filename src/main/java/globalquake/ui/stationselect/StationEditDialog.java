package globalquake.ui.stationselect;

import globalquake.database.Channel;
import globalquake.database.Station;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.awt.event.ActionEvent;

public class StationEditDialog extends JDialog {
    public StationEditDialog(StationSelectFrame stationSelectFrame, Station selectedStation) {
        super(stationSelectFrame);
        setTitle(selectedStation.toString());
        setFont(new Font("Calibri", Font.BOLD, 14));

        setResizable(false);
        setModal(true);
        setLayout(new BorderLayout());

        JTextArea textAreaInfo = createInfoTextArea(selectedStation);

        add(new JScrollPane(textAreaInfo), BorderLayout.NORTH);

        JPanel channelSelectPanel = new JPanel();

        JComboBox<Channel> channelJComboBox = new JComboBox<>();
        channelJComboBox.addItem(null);
        selectedStation.getChannels().forEach(channelJComboBox::addItem);
        channelJComboBox.setSelectedItem(selectedStation.getSelectedChannel());

        channelJComboBox.addItemListener(itemEvent -> selectedStation.setSelectedChannel((Channel) channelJComboBox.getSelectedItem()));

        channelSelectPanel.add(new JLabel("Selected Channel:"));
        channelSelectPanel.add(channelJComboBox);
        add(channelSelectPanel, BorderLayout.CENTER);

        JButton doneButton = new JButton("Done");
        doneButton.addActionListener(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                StationEditDialog.this.dispose();
            }
        });

        JPanel buttonPanel = new JPanel();
        buttonPanel.setBorder(new EmptyBorder(5,5,5,5));
        buttonPanel.add(doneButton);
        add(buttonPanel, BorderLayout.SOUTH);

        pack();
        setLocationRelativeTo(stationSelectFrame);
        setVisible(true);
    }

    private JTextArea createInfoTextArea(Station selectedStation) {
        JTextArea textAreaInfo = new JTextArea();
        textAreaInfo.setEditable(false);
        textAreaInfo.append("Network Code: %s\n".formatted(selectedStation.getNetwork().getNetworkCode()));
        textAreaInfo.append("Network Description:\n    %s\n".formatted(selectedStation.getNetwork().getDescription().trim()));
        textAreaInfo.append("Station Code: %s\n".formatted(selectedStation.getStationCode()));
        textAreaInfo.append("Station Site:\n    %s\n".formatted(selectedStation.getStationSite().trim()));
        textAreaInfo.append("Elevation: %.1fm\n".formatted(selectedStation.getAlt()));
        textAreaInfo.append("Latitude: %f\n".formatted(selectedStation.getLatitude()));
        textAreaInfo.append("Longitude: %f\n".formatted(selectedStation.getLongitude()));
        textAreaInfo.setFont(getFont());
        textAreaInfo.setColumns(22);
        return textAreaInfo;
    }
}

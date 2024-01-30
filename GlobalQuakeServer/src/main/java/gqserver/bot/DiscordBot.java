package gqserver.bot;

import globalquake.core.GlobalQuake;
import globalquake.core.earthquake.EarthquakeAnalysis;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.events.GlobalQuakeEventListener;
import globalquake.core.events.specific.QuakeCreateEvent;
import globalquake.core.events.specific.QuakeRemoveEvent;
import globalquake.core.events.specific.QuakeReportEvent;
import globalquake.core.events.specific.QuakeUpdateEvent;
import globalquake.core.geo.DistanceUnit;
import globalquake.core.intensity.IntensityScales;
import globalquake.core.intensity.Level;
import globalquake.utils.GeoUtils;
import net.dv8tion.jda.api.EmbedBuilder;
import net.dv8tion.jda.api.JDA;
import net.dv8tion.jda.api.JDABuilder;
import net.dv8tion.jda.api.entities.Message;
import net.dv8tion.jda.api.entities.Role;
import net.dv8tion.jda.api.entities.channel.concrete.TextChannel;
import net.dv8tion.jda.api.events.session.ReadyEvent;
import net.dv8tion.jda.api.hooks.ListenerAdapter;
import net.dv8tion.jda.api.requests.GatewayIntent;

import javax.imageio.ImageIO;

import globalquake.core.Settings;
import net.dv8tion.jda.api.utils.FileUpload;
import org.jetbrains.annotations.NotNull;
import org.tinylog.Logger;

import java.awt.*;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class DiscordBot extends ListenerAdapter {

    private static final String TAG = "Discord Bot";
    private static final String VERSION = "0.3";
    private static final String PING_M4 = "Ping 4.0+";
    private static final String PING_M5 = "Ping 5.0+";
    private static final String PING_M6 = "Ping 6.0+";
    private static final String PING_M7 = "Ping 7.0+";
    private static JDA jda;

    private static final Map<Earthquake, Message> lastMessages = new HashMap<>();

    public static void init() {
        jda = JDABuilder.createDefault(Settings.discordBotToken).enableIntents(GatewayIntent.GUILD_MESSAGES)
                .addEventListeners(new DiscordBot())
                .build();

        GlobalQuake.instance.getEventHandler().registerEventListener(new GlobalQuakeEventListener() {
            @Override
            public void onQuakeCreate(QuakeCreateEvent event) {
                sendQuakeCreateInfo(event.earthquake());
            }

            @Override
            public void onQuakeUpdate(QuakeUpdateEvent event) {
                sendQuakeUpdateInfo(event.earthquake());
            }

            @Override
            public void onQuakeRemove(QuakeRemoveEvent event) {
                sendQuakeRemoveInfo(event.earthquake());
            }

            @Override
            public void onQuakeReport(QuakeReportEvent event) {
                sendQuakeReportInfo(event);
            }
        });

        Executors.newSingleThreadScheduledExecutor().scheduleAtFixedRate(DiscordBot::removeOld, 0, 1, TimeUnit.MINUTES);
    }

    private static void removeOld() {
        lastMessages.entrySet().removeIf(kv -> EarthquakeAnalysis.shouldRemove(kv.getKey(), -60 * 10));
    }

    private static void sendQuakeReportInfo(QuakeReportEvent event) {
        TextChannel channel = getChannel();

        if (channel == null) {
            return;
        }

        EmbedBuilder builder = new EmbedBuilder();
        builder.setAuthor("Final Report (#%d)".formatted(event.earthquake().getRevisionID()));

        builder.setImage("attachment://map.png");
        builder.setThumbnail("attachment://int.png");
        createDescription(builder, event.earthquake(), channel);

        ByteArrayOutputStream baosMap = new ByteArrayOutputStream();
        ByteArrayOutputStream baosInt = new ByteArrayOutputStream();
        try {
            ImageIO.write(event.map(), "png", baosMap);
            ImageIO.write(event.intensities(), "png", baosInt);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        Message lastMessage = lastMessages.getOrDefault(event.earthquake(), null);
        if (lastMessage != null) {
            lastMessage.editMessageEmbeds(builder.build())
                    .setFiles(FileUpload.fromData(baosMap.toByteArray(), "map.png"),
                            FileUpload.fromData(baosInt.toByteArray(), "int.png"))
                    .queue();
        } else {
            channel.sendMessageEmbeds(builder.build())
                    .addFiles(FileUpload.fromData(baosMap.toByteArray(), "map.png"),
                            FileUpload.fromData(baosInt.toByteArray(), "int.png"))
                    .queue(message -> lastMessages.put(event.earthquake(), message));
        }

    }

    private static TextChannel getChannel() {
        if (jda == null) {
            return null;
        }

        var guild = jda.getGuildById(Settings.discordBotGuildID);

        if (guild == null) {
            Logger.tag(TAG).error("Unable to find the guild!");
            return null;
        }

        var channel = guild.getTextChannelById(Settings.discordBotChannelID);
        if (channel == null) {
            Logger.tag(TAG).error("Unable to find the channel!");
        }

        return channel;
    }

    private static void sendQuakeRemoveInfo(Earthquake earthquake) {
        TextChannel channel = getChannel();

        if (channel == null) {
            return;
        }

        EmbedBuilder builder = new EmbedBuilder();
        builder.setTitle("M%.1f %s".formatted(earthquake.getMag(), earthquake.getRegion()));
        builder.setAuthor("CANCELED");

        updateMessage(earthquake, builder, channel);
    }

    private static void sendQuakeUpdateInfo(Earthquake earthquake) {
        TextChannel channel = getChannel();

        if (channel == null) {
            return;
        }

        EmbedBuilder builder = new EmbedBuilder();
        builder.setAuthor("Revision #%d".formatted(earthquake.getRevisionID()));
        createDescription(builder, earthquake, channel);

        updateMessage(earthquake, builder, channel);
    }

    private static void updateMessage(Earthquake earthquake, EmbedBuilder builder, TextChannel channel) {
        Message lastMessage = lastMessages.getOrDefault(earthquake, null);

        if (lastMessage != null) {
            lastMessage.editMessageEmbeds(builder.build()).queue();
        } else {
            channel.sendMessageEmbeds(builder.build()).queue(message -> lastMessages.put(earthquake, message));
        }
    }

    private static void sendQuakeCreateInfo(Earthquake earthquake) {
        TextChannel channel = getChannel();

        if (channel == null) {
            return;
        }

        EmbedBuilder builder = new EmbedBuilder();
        builder.setAuthor("New Event");
        createDescription(builder, earthquake, channel);

        updateMessage(earthquake, builder, channel);
    }

    private static void createDescription(EmbedBuilder builder, Earthquake earthquake, TextChannel channel) {
        builder.setTitle("M%.1f %s".formatted(
                earthquake.getMag(),
                earthquake.getRegion()));

        double pga = GeoUtils.getMaxPGA(earthquake.getLat(), earthquake.getLon(), earthquake.getDepth(), earthquake.getMag());

        builder.setDescription(
                tagRoles(channel, earthquake) + "\n" +
                        "Depth: %.1fkm / %.1fmi\n".formatted(earthquake.getDepth(), earthquake.getDepth() * DistanceUnit.MI.getKmRatio()) +
                        "MMI: %s / Shindo: %s\n".formatted(formatLevel(IntensityScales.MMI.getLevel(pga)),
                                formatLevel(IntensityScales.SHINDO.getLevel(pga))) +
                        "Time: %s\n".formatted(Settings.formatDateTime(Instant.ofEpochMilli(earthquake.getOrigin()))) +
                        "Quality: %s (%d stations)".formatted(earthquake.getCluster().getPreviousHypocenter().quality.getSummary(), earthquake.getCluster().getAssignedEvents().size())
        );

        Level level = IntensityScales.getIntensityScale().getLevel(pga);
        Color levelColor = level == null ? Color.gray : level.getColor();

        builder.setFooter("Created at %s with GQ Bot v%s".formatted(Settings.formatDateTime(Instant.ofEpochMilli(earthquake.getCreatedAt())), VERSION));

        builder.setColor(levelColor);
    }

    private static CharSequence tagRoles(TextChannel channel, Earthquake earthquake) {
        java.util.List<Role> rolesToPing = new ArrayList<>();
        if (earthquake.getMag() >= 4.0) {
            rolesToPing.addAll(channel.getGuild().getRolesByName(PING_M4, true));
        }
        if (earthquake.getMag() >= 5.0) {
            rolesToPing.addAll(channel.getGuild().getRolesByName(PING_M5, true));
        }
        if (earthquake.getMag() >= 6.0) {
            rolesToPing.addAll(channel.getGuild().getRolesByName(PING_M6, true));
        }
        if (earthquake.getMag() >= 7.0) {
            rolesToPing.addAll(channel.getGuild().getRolesByName(PING_M7, true));
        }

        StringBuilder stringBuilder = new StringBuilder();
        for (Role role : rolesToPing) {
            stringBuilder.append(role.getAsMention());
        }

        return stringBuilder.toString();
    }

    private static String formatLevel(Level level) {
        if (level == null) {
            return "-";
        } else {
            return level.toString();
        }
    }

    @Override
    public void onReady(@NotNull ReadyEvent event) {
        TextChannel channel = getChannel();

        if (channel == null) {
            return;
        }

        EmbedBuilder builder = new EmbedBuilder();
        builder.setTitle("GlobalQuake BOT v%s".formatted(VERSION));
        builder.setDescription("Starting up...");

        channel.sendMessageEmbeds(builder.build()).queue();

        Logger.info("Discord bot ready");
    }

}
